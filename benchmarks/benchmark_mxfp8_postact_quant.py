"""Cold-L2 benchmark for fused gated-GEMM post-activation quantization.

Compares:
  baseline: MXFP8 up-projection -> BF16 postact, then BF16 -> FP8 quant
  fused:    MXFP8 up-projection -> FP8 postact + scales in the epilogue

The L2 flush is issued on the same CUDA stream immediately before each timed
invocation and is not included in the elapsed time.
"""

import argparse
import math
import statistics
from dataclasses import replace

import torch

from quack.gemm_blockscaled_sm90 import (
    mxfp8_gemm_gated_postact_quant_sm90,
    mxfp8_gemm_gated_tuned_sm90,
    quantize_act,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig
from quack.quant import _blockwise_quant


CONFIG = GemmConfig(
    tile_m=128,
    tile_n=256,
    cluster_m=2,
    cluster_n=1,
    pingpong=False,
    is_dynamic_persistent=False,
)

FUSED_CONFIG = GemmConfig(
    tile_m=128,
    tile_n=256,
    epi_tile_n=256,
    cluster_m=1,
    cluster_n=1,
    pingpong=False,
    is_dynamic_persistent=False,
)


def measure_cold_l2(fn, flush, warmup=20, repeats=100):
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        flush.zero_()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1_000.0)
    return statistics.median(samples), min(samples)


def measure_pair_cold_l2(lhs, rhs, flush, warmup=20, repeats=100):
    """Interleave two paths, reversing their order every sample to limit clock bias."""
    for _ in range(warmup):
        flush.zero_()
        lhs()
        flush.zero_()
        rhs()
    torch.cuda.synchronize()
    samples = ([], [])
    for repeat in range(repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for which in order:
            flush.zero_()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            (lhs if which == 0 else rhs)()
            end.record()
            end.synchronize()
            samples[which].append(begin.elapsed_time(end) * 1_000.0)
    return tuple((statistics.median(x), min(x)) for x in samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--allow-non-bitwise",
        action="store_true",
        help="Report fused postact differences instead of requiring bitwise identity",
    )
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--fused-cluster-m", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--flush-l2-multiple",
        type=int,
        default=3,
        help="Size of the cache-thrashing buffer as a multiple of device L2",
    )
    args = parser.parse_args()
    fused_config = replace(FUSED_CONFIG, cluster_m=args.fused_cluster_m)

    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(20260723)
    device = torch.device("cuda")
    rows = args.tokens * args.top_k
    assert rows % args.experts == 0
    assert args.hidden_size % 128 == 0
    assert args.intermediate_size % 128 == 0
    lengths = (rows // args.experts,) * args.experts
    experts = args.experts
    k = args.hidden_size
    postact_n = args.intermediate_size
    preact_n = 2 * postact_n
    cu = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)

    # Match fp8_trace.py: the GG gathers T source rows into T * top_k routed rows.
    source = torch.randn((args.tokens, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    weight = (
        torch.randn((experts, preact_n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    )
    qa, dense_sfa = quantize_act(source)
    qb, sfb = quantize_weight_sm90(weight)
    a_idx = torch.arange(rows, dtype=torch.int32, device=device).remainder_(args.tokens)
    # This column-major view matches gather_padded_sfa's dQaccum scale layout.
    sfa = torch.empty(k // 128, rows, dtype=torch.float32, device=device).mT
    sfa.copy_(dense_sfa[a_idx.long()])

    preact = torch.empty((rows, preact_n), device=device, dtype=torch.bfloat16)
    postact = torch.empty((rows, postact_n), device=device, dtype=torch.bfloat16)
    qpostact = torch.empty_like(postact, dtype=torch.float8_e4m3fn)
    postact_scale = torch.empty(
        (postact_n // 128, rows), device=device, dtype=torch.float32
    ).mT
    props = torch.cuda.get_device_properties(device)
    flush_bytes = args.flush_l2_multiple * props.L2_cache_size + 1
    flush = torch.empty(flush_bytes, device=device, dtype=torch.uint8)

    def gemm():
        mxfp8_gemm_gated_tuned_sm90.fn(
            qa,
            qb.mT,
            sfa,
            sfb.mT,
            preact,
            postact,
            activation="swiglu",
            cu_seqlens_m=cu,
            A_idx=a_idx,
            config=CONFIG,
        )

    def quant():
        _blockwise_quant(postact, qpostact, postact_scale, None, 128)

    def baseline():
        gemm()
        quant()

    def fused():
        mxfp8_gemm_gated_postact_quant_sm90(
            qa,
            qb.mT,
            sfa,
            sfb.mT,
            preact,
            qpostact,
            postact_scale,
            cu,
            A_idx=a_idx,
            config=fused_config,
        )

    # Compile all paths before timing.
    baseline()
    preact_ref = preact.clone()
    qpostact_ref = qpostact.clone()
    postact_scale_ref = postact_scale.clone()
    fused()
    torch.cuda.synchronize()
    assert torch.equal(preact, preact_ref), "fused preact is not bitwise identical"
    if not torch.equal(qpostact, qpostact_ref):
        print(
            "FP8 mismatches:",
            int((qpostact.float() != qpostact_ref.float()).sum()),
            "of",
            qpostact.numel(),
        )
    if not torch.equal(postact_scale, postact_scale_ref):
        scale_bad = (postact_scale != postact_scale_ref).nonzero()
        first_bad = scale_bad[0]
        print(
            "scale mismatches:",
            int((postact_scale != postact_scale_ref).sum()),
            "of",
            postact_scale.numel(),
            "max abs:",
            float((postact_scale - postact_scale_ref).abs().max()),
            "first:",
            first_bad.tolist(),
            float(postact_scale[first_bad[0], first_bad[1]]),
            float(postact_scale_ref[first_bad[0], first_bad[1]]),
        )
        print("fused scale rows 0:4:", postact_scale[:4].cpu())
        print("reference scale rows 0:4:", postact_scale_ref[:4].cpu())
    if not args.allow_non_bitwise:
        assert torch.equal(
            qpostact, qpostact_ref
        ), "fused FP8 postact is not bitwise identical"
        assert torch.equal(
            postact_scale, postact_scale_ref
        ), "fused postact scales are not bitwise identical"
    gemm_median, gemm_min = measure_cold_l2(gemm, flush, args.warmup, args.repeats)
    quant_median, quant_min = measure_cold_l2(quant, flush, args.warmup, args.repeats)
    (total_median, total_min), (fused_median, fused_min) = measure_pair_cold_l2(
        baseline, fused, flush, args.warmup, args.repeats
    )
    print(
        f"shape: tokens={args.tokens}, top_k={args.top_k}, experts={experts}, "
        f"gathered_rows={rows}, K={k}, "
        f"preact_N={preact_n}, postact_N={postact_n}"
    )
    print(f"fused config: cluster={args.fused_cluster_m}x1")
    print(
        f"cold L2: {props.L2_cache_size / 2**20:.1f} MiB cache, "
        f"{flush_bytes / 2**20:.1f} MiB flush before every sample"
    )
    print(f"grouped_gemm_gated_bf16: median={gemm_median:.2f} us min={gemm_min:.2f} us")
    print(f"separate_postact_quant: median={quant_median:.2f} us min={quant_min:.2f} us")
    print(f"baseline_combined: median={total_median:.2f} us min={total_min:.2f} us")
    print(f"fused_postact_quant: median={fused_median:.2f} us min={fused_min:.2f} us")
    print(f"speedup: median={total_median / fused_median:.3f}x min={total_min / fused_min:.3f}x")


if __name__ == "__main__":
    main()
