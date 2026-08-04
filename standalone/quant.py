# Copyright (c) 2025, Quack authors.

"""Standalone blockwise FP8 quantization.

This module intentionally has no dependency on the :mod:`quack` package.  Its runtime
dependencies are PyTorch, CUDA Python, and NVIDIA CUTLASS DSL.
"""

import functools
import math
from functools import partial
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Boolean, Float32, Int32, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import arith, llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op
from torch import Tensor

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


@dsl_user_op
def _copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    loc=None,
    ip=None,
) -> None:
    """Copy one vector between global and register memory."""
    num_copy_bits = const_expr(min(128, src.shape[0][0] * src.element_type.width))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=num_copy_bits
    )
    cute.copy(atom, src, dst, pred=pred, loc=loc, ip=ip)


def _tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int,
) -> cute.TiledCopy:
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=num_copy_elems * dtype.width,
    )
    thread_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    value_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)


@cute.jit
def _predicate_k(coords: cute.Tensor, limit: Int32) -> cute.Tensor:
    pred = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(coords, mode=[0, 1]),
                cute.size(coords, mode=[1]),
                cute.size(coords, mode=[2]),
            ),
            stride=(cute.size(coords, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(pred.shape[0]):
        for rest_k in cutlass.range_constexpr(pred.shape[2]):
            pred[rest_v, 0, rest_k] = cute.elem_less(coords[(0, rest_v), 0, rest_k][1], limit)
    return pred


@cute.jit
def _row_max(x: cute.TensorSSA, threads_per_row: cutlass.Constexpr[int]) -> Float32:
    value = x.reduce(cute.ReductionOp.MAX, init_val=1e-5, reduction_profile=0)
    return cute.arch.warp_reduction(value, cute.arch.fmax, threads_in_group=threads_per_row)


def _ceil_log2(x):
    xm1 = (Int32(x) - 1).ir_value()
    return Int32(32) - Int32(llvm.intr_ctlz(xm1, False))


class _FastDivmod:
    """Host-precomputed unsigned divmod used to address the 2-D scale tensor."""

    def __init__(self, divisor):
        if isinstance(divisor, int):
            divisor = Int32(divisor)
        self.divisor = divisor
        shift = cutlass.max(_ceil_log2(divisor) - 1, Int32(0))
        power = Uint64(Uint32(1) << Uint32(shift))
        numerator = Uint64(0x100000000) * power
        magic = (numerator + Uint64(Uint32(divisor)) - 1) // Uint64(Uint32(divisor))
        self.magic = Uint32(magic & 0xFFFFFFFF)
        self.shift = Uint32(shift)

    def __rdivmod__(self, dividend):
        quotient = Uint32(cute.arch.mul_hi(Uint32(dividend), self.magic)) >> self.shift
        quotient = Int32(cutlass.select_(self.magic == Uint32(0), Int32(dividend), Int32(quotient)))
        return quotient, Int32(dividend) - quotient * Int32(self.divisor)

    def __extract_mlir_values__(self):
        values = []
        self._value_counts = []
        for obj in (self.magic, self.shift, self.divisor):
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._value_counts.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        new = object.__new__(_FastDivmod)
        for name, count in zip(("magic", "shift", "divisor"), self._value_counts):
            old = getattr(self, name)
            setattr(new, name, cutlass.new_from_mlir_values(old, values[:count]))
            values = values[count:]
        new._value_counts = self._value_counts
        return new


class BlockwiseQuant:
    """CuTe DSL kernel for power-of-two-scaled blockwise FP8 quantization."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        block_size: int,
        threads_per_row: int = 8,
        num_threads: int = 128,
    ):
        self.dtype = dtype
        self.block_size = block_size
        self.threads_per_row = threads_per_row
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        scale: cute.Tensor,
        scale_row_idx: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        largest_width = const_expr(max(x.element_type.width, out.element_type.width))
        vecsize = math.gcd(self.block_size, 128 // largest_width)
        blocks_n = cute.ceil_div(self.block_size // vecsize, self.threads_per_row)
        tiler_mn = (
            self.num_threads // self.threads_per_row,
            vecsize * blocks_n * self.threads_per_row,
        )
        tiled_copy = _tiled_copy_2d(self.dtype, self.threads_per_row, self.num_threads, vecsize)
        self.kernel(
            x,
            out,
            scale,
            scale_row_idx,
            tiler_mn,
            tiled_copy,
            _FastDivmod(scale.shape[1]),
        ).launch(
            grid=[cute.ceil_div(x.shape[0], tiler_mn[0]), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        scale: cute.Tensor,
        scale_row_idx: cute.Tensor | None,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        blocks_n_divmod: _FastDivmod,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        coords = cute.make_identity_tensor(x.shape)
        gx, gout, cx = [cute.local_tile(tensor, tiler_mn, (bidx, 0)) for tensor in (x, out, coords)]
        thread_copy = tiled_copy.get_slice(tidx)
        tx_gx = thread_copy.partition_S(gx)
        tx_gout = thread_copy.partition_D(gout)
        tx_cx = thread_copy.partition_S(cx)[(0, None), None, None]
        tx_rx, tx_rout = [cute.make_rmem_tensor_like(t) for t in (tx_gx, tx_gout)]
        pred = (
            _predicate_k(thread_copy.partition_S(cx), limit=x.shape[1])
            if not const_expr(x.shape[1] == tiler_mn[1])
            else None
        )
        copy = partial(_copy, pred=pred)
        row = tx_cx[0][0]
        if row < x.shape[0]:
            copy(tx_gx, tx_rx)
        values = tx_rx.load().to(cute.Float32)
        abs_values = cute.make_rmem_tensor_like(tx_rx, cute.Float32)
        for i in cutlass.range_constexpr(const_expr(cute.size(abs_values))):
            abs_values[i] = mlir_math.absf(Float32(tx_rx[i]))
        amax = _row_max(abs_values.load(), self.threads_per_row)
        quant_scale = FP8_MAX / amax
        scale_bits = llvm.bitcast(T.i32(), quant_scale) & 0xFF800000
        quant_scale = arith.bitcast(T.f32(), scale_bits)
        tx_rout.store((values * quant_scale).to(tx_rout.element_type))
        if row < x.shape[0]:
            copy(tx_rout, tx_gout)
        if tx_cx[0][1] == 0 and row < x.shape[0]:
            m, block_j = divmod(row, blocks_n_divmod)
            if const_expr(scale_row_idx is not None):
                m = scale_row_idx[m]
            scale[m, block_j] = 1.0 / quant_scale


def _fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    if leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


@functools.cache
def _compile_quant(
    dtype,
    out_dtype,
    block_size,
    scale_transpose=False,
    has_scatter=False,
):
    batch = cute.sym_int()
    alignment = math.gcd(block_size, 128 // dtype.width, 128 // out_dtype.width)
    x = _fake_tensor(dtype, (batch, block_size), alignment)
    out = _fake_tensor(out_dtype, (batch, block_size), alignment)
    scale = _fake_tensor(
        Float32,
        (cute.sym_int(), cute.sym_int()),
        leading_dim=0 if scale_transpose else 1,
    )
    row_idx = _fake_tensor(cutlass.Int32, (cute.sym_int(),)) if has_scatter else None
    return cute.compile(
        BlockwiseQuant(dtype, block_size),
        x,
        out,
        scale,
        row_idx,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _launch_quant(
    x: Tensor,
    out: Tensor,
    scale: Tensor,
    scale_row_idx: Tensor | None,
    block_size: int,
) -> None:
    compiled = _compile_quant(
        _TORCH_TO_CUTE_DTYPE[x.dtype],
        _TORCH_TO_CUTE_DTYPE[out.dtype],
        block_size,
        scale.stride(-1) != 1,
        scale_row_idx is not None,
    )
    compiled(x, out, scale, scale_row_idx)


@torch.library.custom_op(
    "standalone_cutedsl::_blockwise_quant",
    mutates_args=("out", "scale"),
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor(a1!) out, Tensor(a2!) scale, "
        "Tensor? scale_row_idx, int block_size) -> ()"
    ),
)
def _blockwise_quant(
    x: Tensor,
    out: Tensor,
    scale: Tensor,
    scale_row_idx: Tensor | None,
    block_size: int,
) -> None:
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.shape[1] % block_size == 0
    _launch_quant(
        x.reshape(-1, block_size),
        out.reshape(-1, block_size),
        scale,
        scale_row_idx,
        block_size,
    )


def blockwise_quant(
    src: Tensor,
    block_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_transpose: bool = False,
    scale_row_idx: Tensor | None = None,
    scale_rows: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize the last dimension in blocks and return ``(quantized, scale)``."""
    n = src.shape[-1]
    original_shape = src.shape
    src_2d = src.view(-1, n)
    assert n % block_size == 0
    if scale_row_idx is not None:
        assert scale_rows is not None
    rows = scale_rows if scale_row_idx is not None else src_2d.shape[0]
    out = torch.empty_like(src_2d, dtype=quant_dtype)
    if scale_transpose:
        scale = torch.empty(
            n // block_size, rows, device=src.device, dtype=torch.float32
        ).transpose(0, 1)
    else:
        scale = torch.empty(rows, n // block_size, device=src.device, dtype=torch.float32)
    _blockwise_quant(src_2d, out, scale, scale_row_idx, block_size)
    return out.view(original_shape), scale.view(*original_shape[:-1], n // block_size)


def quant_ref(
    src: Tensor,
    block_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    min_scale: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """PyTorch reference for the standalone kernel."""
    src_f32 = src.reshape(-1, block_size).float()
    amax = src_f32.abs().amax(dim=1).clamp(min=min_scale)
    quant_scale = torch.tensor(torch.finfo(quant_dtype).max, device=src.device) / amax
    quant_scale = (quant_scale.view(torch.int32) & 0xFF800000).view(torch.float32)
    quantized = (src_f32 * quant_scale[:, None]).to(quant_dtype)
    return (
        quantized.reshape(src.shape),
        (1.0 / quant_scale).reshape(*src.shape[:-1], src.shape[-1] // block_size),
    )


@triton.jit
def _triton_quant_kernel(
    src,
    out,
    scale,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """One-dimensional grid: each program quantizes several adjacent blocks."""
    first_block = tl.program_id(0) * BLOCKS_PER_PROGRAM
    block_ids = first_block + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_offsets = block_ids[:, None]
    element_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    offsets = block_offsets * BLOCK_SIZE + element_offsets
    mask = block_offsets < num_blocks
    values = tl.load(src + offsets, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-5)
    quant_scale = 448.0 / amax
    scale_bits = quant_scale.to(tl.uint32, bitcast=True) & 0xFF800000
    quant_scale = scale_bits.to(tl.float32, bitcast=True)
    tl.store(out + offsets, values * quant_scale[:, None], mask=mask)
    tl.store(scale + block_ids, 1.0 / quant_scale, mask=block_ids < num_blocks)


def triton_quant(
    src: Tensor,
    block_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    blocks_per_program: int = 16,
) -> tuple[Tensor, Tensor]:
    """Blockwise FP8 quantization using a one-dimensional Triton kernel."""
    assert src.is_cuda
    assert src.is_contiguous()
    assert src.numel() % block_size == 0
    assert quant_dtype == torch.float8_e4m3fn
    assert blocks_per_program in (1, 2, 4, 8, 16)
    out = torch.empty_like(src, dtype=quant_dtype)
    scale = torch.empty(
        *src.shape[:-1],
        src.shape[-1] // block_size,
        device=src.device,
        dtype=torch.float32,
    )
    num_blocks = src.numel() // block_size
    _triton_quant_kernel[(triton.cdiv(num_blocks, blocks_per_program),)](
        src,
        out,
        scale,
        num_blocks,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        num_warps=8,
    )
    return out, scale


def blockwise_dequant(
    src: Tensor,
    scale: Tensor,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize blockwise FP8 values using their reciprocal scales."""
    values = src.reshape(-1, block_size).float() * scale.reshape(-1, 1)
    return values.to(out_dtype).reshape(src.shape)
