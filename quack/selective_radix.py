# ruff: noqa: N803, PLR0912, PLR0915, PLR1730, PLR2004, TC002
# mypy: ignore-errors
"""CuTe DSL implementation of local-expert selective radix routing."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from quack import utils
from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity
from quack.dsl import cute_op

_THREADS = 256
_WARPS = _THREADS // cute.arch.WARP_SIZE
_ITEMS = 4
_OFFSET_ITEMS = 3
_CHUNK = _THREADS * _ITEMS
_RADIX_BINS = 256
_CUSTOM_BINS = _RADIX_BINS + 3
_MIN_NORMAL_KEY = 0x00800000
_MAX_LOCAL_EXPERTS = 256
_MATRIX_NDIM = 2

_WORK_COUNT = 0
_RANK = 1
_BOUNDARY = 2
_NO_DROP = 3
_POSITIVE_COUNT = 4
_THRESHOLD = 5
_TOTAL_GREATER = 6
_EQUAL_NEEDED = 7
_APPROXIMATE = 9
_REFINED_COUNT = 10
_STATE_SIZE = 11


@cute.jit
def _block_exclusive_sum(value: Int32, warp_offsets: cute.Tensor) -> Int32:
    """Return an exclusive CTA sum for one value per thread."""
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    inclusive = utils.warp_prefix_sum(value, lane)
    if lane == cute.arch.WARP_SIZE - 1:
        warp_offsets[warp] = inclusive
    cute.arch.sync_threads()

    if warp == 0:
        warp_value = Int32(0)
        if lane < _WARPS:
            warp_value = Int32(warp_offsets[lane])
        warp_inclusive = utils.warp_prefix_sum(warp_value, lane)
        if lane < _WARPS:
            warp_offsets[lane] = warp_inclusive - warp_value
    cute.arch.sync_threads()
    return inclusive - value + Int32(warp_offsets[warp])


class SelectiveRadix:
    """Specialized selective-radix threshold and metadata construction."""

    def __init__(
        self,
        columns: int,
        local_start: int,
        local_experts: int,
        capacity: int,
        max_packed_rows: int,
    ) -> None:
        self.columns = columns
        self.local_start = local_start
        self.local_end = local_start + local_experts
        self.local_experts = local_experts
        self.capacity = capacity
        self.max_packed_rows = max_packed_rows

    @cute.jit
    def __call__(
        self,
        mProbabilities: cute.Tensor,
        mExperts: cute.Tensor,
        mHistogram: cute.Tensor,
        mState: cute.Tensor,
        mKeysA: cute.Tensor,
        mKeysB: cute.Tensor,
        mGreaterCounts: cute.Tensor,
        mEqualCounts: cute.Tensor,
        mGreaterOffsets: cute.Tensor,
        mEqualOffsets: cute.Tensor,
        mSelectedIndices: cute.Tensor,
        mSelectedValid: cute.Tensor,
        mSelectedExperts: cute.Tensor,
        mLocalPositions: cute.Tensor,
        mTokenIds: cute.Tensor,
        mTopkSlots: cute.Tensor,
        mExpertCounts: cute.Tensor,
        mAlignedCounts: cute.Tensor,
        mExpertOffsets: cute.Tensor,
        mPackedIdx: cute.Tensor,
        mPackedAssignment: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        num_elements = mProbabilities.shape[0] * self.columns
        blocks = cute.ceil_div(num_elements, _CHUNK)
        histogram_grid = [blocks, 1, 1]

        self.custom_histogram_kernel(
            mProbabilities,
            mExperts,
            mHistogram,
            mState,
        ).launch(grid=histogram_grid, block=[_THREADS, 1, 1], stream=stream)
        self.choose_custom_boundary_kernel(mHistogram, mState).launch(
            grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream
        )
        self.compact_custom_boundary_kernel(
            mProbabilities,
            mExperts,
            mKeysB,
            mState,
        ).launch(grid=histogram_grid, block=[_THREADS, 1, 1], stream=stream)
        self.refine_and_compact_kernel(mKeysB, mKeysA, mState).launch(
            grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream
        )

        self.selection_counts_kernel(
            mProbabilities,
            mExperts,
            mGreaterCounts,
            mEqualCounts,
            mState,
            mSelectedIndices,
            mSelectedValid,
            mExpertCounts,
        ).launch(grid=histogram_grid, block=[_THREADS, 1, 1], stream=stream)
        self.selection_offsets_kernel(
            mGreaterCounts,
            mEqualCounts,
            mGreaterOffsets,
            mEqualOffsets,
            mState,
        ).launch(grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream)
        self.selection_scatter_kernel(
            mProbabilities,
            mExperts,
            mGreaterOffsets,
            mEqualOffsets,
            mState,
            mSelectedIndices,
            mSelectedValid,
            mSelectedExperts,
            mLocalPositions,
            mTokenIds,
            mTopkSlots,
            mExpertCounts,
        ).launch(grid=histogram_grid, block=[_THREADS, 1, 1], stream=stream)
        self.finalize_counts_offsets_kernel(mExpertCounts, mAlignedCounts, mExpertOffsets).launch(
            grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream
        )
        self.finalize_mapping_scatter_kernel(
            mSelectedValid,
            mSelectedExperts,
            mLocalPositions,
            mExpertOffsets,
            mPackedIdx,
            mPackedAssignment,
        ).launch(
            grid=[cute.ceil_div(self.capacity, _THREADS), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def custom_histogram_kernel(
        self,
        mProbabilities: cute.Tensor,
        mExperts: cute.Tensor,
        mHistogram: cute.Tensor,
        mState: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        histogram = smem.allocate_tensor(
            Int32,
            cute.make_layout((_CUSTOM_BINS,)),
            byte_alignment=16,
        )
        for item in cutlass.range_constexpr(cute.ceil_div(_CUSTOM_BINS, _THREADS)):
            bin_index = tidx + item * _THREADS
            if bin_index < _CUSTOM_BINS:
                histogram[bin_index] = Int32(0)
        cute.arch.sync_threads()

        local_positive = Int32(0)
        num_elements = mProbabilities.shape[0] * self.columns
        for item in cutlass.range_constexpr(_ITEMS):
            index = Int32(bidx * _CHUNK + tidx * _ITEMS + item)
            if index < num_elements:
                row = index // self.columns
                column = index - row * self.columns
                expert = Int32(mExperts[row, column])
                probability = Float32(mProbabilities[row, column])
                key = llvm.bitcast(T.i32(), probability)
                if (
                    (expert >= self.local_start)
                    & (expert < self.local_end)
                    & (probability > 0.0)
                    & (key >= _MIN_NORMAL_KEY)
                ):
                    prefix5 = key >> 27
                    digit = Int32(0)
                    if prefix5 > 3:
                        digit = Int32(1)
                    if prefix5 > 5:
                        digit = Int32(2)
                    if prefix5 > 6:
                        digit = Int32(3) + ((key >> 19) & 0xFF)
                    cute.arch.atomic_add(
                        histogram.iterator + digit,
                        Int32(1),
                        scope="cta",
                    )
                    local_positive += 1
        if local_positive != 0:
            cute.arch.atomic_add(
                mState.iterator + _POSITIVE_COUNT,
                local_positive,
                scope="gpu",
            )
        cute.arch.sync_threads()
        for item in cutlass.range_constexpr(cute.ceil_div(_CUSTOM_BINS, _THREADS)):
            bin_index = tidx + item * _THREADS
            if bin_index < _CUSTOM_BINS:
                count = Int32(histogram[bin_index])
                if count != 0:
                    cute.arch.atomic_add(
                        mHistogram.iterator + bin_index,
                        count,
                        scope="gpu",
                    )

    @cute.kernel
    def choose_custom_boundary_kernel(
        self,
        mHistogram: cute.Tensor,
        mState: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        no_drop = Boolean(mState[_POSITIVE_COUNT] <= self.capacity)
        if no_drop:
            if tidx == 0:
                mState[_NO_DROP] = Int32(1)
                mState[_THRESHOLD] = Int32(0)
                mState[_WORK_COUNT] = Int32(0)
        else:
            smem = cutlass.utils.SmemAllocator()
            warp_offsets = smem.allocate_tensor(
                Int32,
                cute.make_layout((_WARPS,)),
                byte_alignment=16,
            )
            counts = [Int32(0) for _ in range(2)]
            thread_count = Int32(0)
            for item in cutlass.range_constexpr(2):
                candidate = _CUSTOM_BINS - 1 - tidx * 2 - item
                if candidate >= 0:
                    counts[item] = Int32(mHistogram[candidate])
                    thread_count += counts[item]
            higher = _block_exclusive_sum(thread_count, warp_offsets)
            rank = Int32(self.capacity)
            if (rank > higher) & (rank <= higher + thread_count):
                boundary = Int32(0)
                found = Boolean(False)
                running = higher
                for item in cutlass.range_constexpr(2):
                    candidate = _CUSTOM_BINS - 1 - tidx * 2 - item
                    if not found:
                        count = counts[item]
                        if rank <= running + count:
                            boundary = Int32(candidate)
                            rank -= running
                            found = Boolean(True)
                        else:
                            running += count
                mState[_BOUNDARY] = boundary
                mState[_RANK] = rank
                mState[_APPROXIMATE] = Int32(boundary < 3)
                mState[_WORK_COUNT] = Int32(0)
                threshold = Int32(0x1FFFFFFF)
                if boundary == 1:
                    threshold = Int32(0x2FFFFFFF)
                elif boundary == 2:
                    threshold = Int32(0x37FFFFFF)
                elif boundary >= 3:
                    threshold = Int32(0x38000000) | ((boundary - 3) << 19)
                mState[_THRESHOLD] = threshold

    @cute.kernel
    def compact_custom_boundary_kernel(
        self,
        mProbabilities: cute.Tensor,
        mExperts: cute.Tensor,
        mOutputKeys: cute.Tensor,
        mState: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        active = Boolean((mState[_NO_DROP] == 0) & (mState[_APPROXIMATE] == 0))
        if active:
            smem = cutlass.utils.SmemAllocator()
            warp_offsets = smem.allocate_tensor(
                Int32,
                cute.make_layout((_WARPS,)),
                byte_alignment=16,
            )
            scalars = smem.allocate_tensor(
                Int32,
                cute.make_layout((2,)),
                byte_alignment=8,
            )
            boundary = Int32(mState[_BOUNDARY])
            num_elements = mProbabilities.shape[0] * self.columns
            keys = [Int32(0) for _ in range(_ITEMS)]
            flags = [Boolean(False) for _ in range(_ITEMS)]
            thread_count = Int32(0)
            for item in cutlass.range_constexpr(_ITEMS):
                index = Int32(bidx * _CHUNK + tidx * _ITEMS + item)
                key = Int32(0)
                valid = Boolean(False)
                if index < num_elements:
                    row = index // self.columns
                    column = index - row * self.columns
                    expert = Int32(mExperts[row, column])
                    probability = Float32(mProbabilities[row, column])
                    loaded_key = llvm.bitcast(T.i32(), probability)
                    valid = Boolean(
                        (expert >= self.local_start)
                        & (expert < self.local_end)
                        & (probability > 0.0)
                        & (loaded_key >= _MIN_NORMAL_KEY)
                    )
                    if valid:
                        key = loaded_key
                keys[item] = key
                prefix5 = key >> 27
                digit = Int32(0)
                if prefix5 > 3:
                    digit = Int32(1)
                if prefix5 > 5:
                    digit = Int32(2)
                if prefix5 > 6:
                    digit = Int32(3) + ((key >> 19) & 0xFF)
                flag = Boolean(valid & (digit == boundary))
                flags[item] = flag
                thread_count += Int32(flag)

            thread_prefix = _block_exclusive_sum(thread_count, warp_offsets)
            if tidx == _THREADS - 1:
                scalars[0] = thread_prefix + thread_count
            cute.arch.sync_threads()
            if tidx == 0:
                scalars[1] = cute.arch.atomic_add(
                    mState.iterator + _WORK_COUNT,
                    Int32(scalars[0]),
                    scope="gpu",
                )
            cute.arch.sync_threads()

            local = Int32(0)
            output_base = Int32(scalars[1]) + thread_prefix
            for item in cutlass.range_constexpr(_ITEMS):
                if flags[item]:
                    mOutputKeys[output_base + local] = keys[item]
                    local += 1

    @cute.kernel
    def refine_and_compact_kernel(
        self,
        mKeys: cute.Tensor,
        mOutputKeys: cute.Tensor,
        mState: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        active = Boolean((mState[_NO_DROP] == 0) & (mState[_APPROXIMATE] == 0))
        if active:
            smem = cutlass.utils.SmemAllocator()
            histogram = smem.allocate_tensor(
                Int32,
                cute.make_layout((_RADIX_BINS,)),
                byte_alignment=16,
            )
            warp_offsets = smem.allocate_tensor(
                Int32,
                cute.make_layout((_WARPS,)),
                byte_alignment=16,
            )
            count = Int32(mState[_WORK_COUNT])
            histogram[tidx] = Int32(0)
            cute.arch.sync_threads()
            for index in cutlass.range(tidx, count, _THREADS, unroll=1):
                key = Int32(mKeys[index])
                cute.arch.atomic_add(
                    histogram.iterator + ((key >> 11) & 0xFF),
                    Int32(1),
                    scope="cta",
                )
            cute.arch.sync_threads()

            boundary = _RADIX_BINS - 1 - tidx
            bucket_count = Int32(histogram[boundary])
            higher = _block_exclusive_sum(bucket_count, warp_offsets)
            rank = Int32(mState[_RANK])
            if (rank > higher) & (rank <= higher + bucket_count):
                mState[_BOUNDARY] = boundary
                mState[_RANK] = rank - higher
                mState[_THRESHOLD] = Int32(mState[_THRESHOLD]) | (boundary << 11)
            cute.arch.sync_threads()

            prefix = Int32(mState[_THRESHOLD])
            for index in cutlass.range(tidx, count, _THREADS, unroll=1):
                key = Int32(mKeys[index])
                if (key >> 11) == (prefix >> 11):
                    output = cute.arch.atomic_add(
                        mState.iterator + _REFINED_COUNT,
                        Int32(1),
                        scope="gpu",
                    )
                    mOutputKeys[output] = key
            cute.arch.sync_threads()
            if tidx == 0:
                mState[_WORK_COUNT] = mState[_REFINED_COUNT]
            cute.arch.sync_threads()

            refined_count = Int32(mState[_WORK_COUNT])
            for pass_index in cutlass.range_constexpr(2):
                tail_shift = (3, 0)[pass_index]
                width = (8, 3)[pass_index]
                bins = 1 << width
                if tidx < bins:
                    histogram[tidx] = Int32(0)
                cute.arch.sync_threads()
                current_prefix = Int32(mState[_THRESHOLD])
                for index in cutlass.range(tidx, refined_count, _THREADS, unroll=1):
                    key = Int32(mOutputKeys[index])
                    if (key >> (tail_shift + width)) == (current_prefix >> (tail_shift + width)):
                        cute.arch.atomic_add(
                            histogram.iterator + ((key >> tail_shift) & (bins - 1)),
                            Int32(1),
                            scope="cta",
                        )
                cute.arch.sync_threads()
                bucket_count = Int32(0)
                boundary = Int32(0)
                if tidx < bins:
                    boundary = bins - 1 - tidx
                    bucket_count = Int32(histogram[boundary])
                higher = _block_exclusive_sum(bucket_count, warp_offsets)
                rank = Int32(mState[_RANK])
                if (rank > higher) & (rank <= higher + bucket_count):
                    mState[_RANK] = rank - higher
                    mState[_THRESHOLD] = Int32(mState[_THRESHOLD]) | (boundary << tail_shift)
                cute.arch.sync_threads()

    @cute.kernel
    def selection_counts_kernel(
        self,
        mProbabilities: cute.Tensor,
        mExperts: cute.Tensor,
        mGreaterCounts: cute.Tensor,
        mEqualCounts: cute.Tensor,
        mState: cute.Tensor,
        mSelectedIndices: cute.Tensor,
        mSelectedValid: cute.Tensor,
        mExpertCounts: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        counts = smem.allocate_tensor(
            Int32,
            cute.make_layout((2,)),
            byte_alignment=8,
        )
        if tidx < 2:
            counts[tidx] = Int32(0)
        cute.arch.sync_threads()

        threshold = Int32(mState[_THRESHOLD])
        no_drop = Boolean(mState[_NO_DROP] != 0)
        local_greater = Int32(0)
        local_equal = Int32(0)
        num_elements = mProbabilities.shape[0] * self.columns
        for item in cutlass.range_constexpr(_ITEMS):
            index = Int32(bidx * _CHUNK + tidx * _ITEMS + item)
            if index < self.capacity:
                mSelectedIndices[index] = Int64(0)
                mSelectedValid[index] = Boolean(False)
            if index < num_elements:
                row = index // self.columns
                column = index - row * self.columns
                expert = Int32(mExperts[row, column])
                probability = Float32(mProbabilities[row, column])
                key = llvm.bitcast(T.i32(), probability)
                valid = Boolean(
                    (expert >= self.local_start)
                    & (expert < self.local_end)
                    & (probability > 0.0)
                    & (key >= _MIN_NORMAL_KEY)
                )
                if valid:
                    if no_drop:
                        local_greater += 1
                    else:
                        if key > threshold:
                            local_greater += 1
                        if key == threshold:
                            local_equal += 1
        if (bidx == 0) & (tidx < self.local_experts):
            mExpertCounts[tidx] = Int32(0)
        if local_greater != 0:
            cute.arch.atomic_add(
                counts.iterator,
                local_greater,
                scope="cta",
            )
        if local_equal != 0:
            cute.arch.atomic_add(
                counts.iterator + 1,
                local_equal,
                scope="cta",
            )
        cute.arch.sync_threads()
        if tidx == 0:
            mGreaterCounts[bidx] = counts[0]
            mEqualCounts[bidx] = counts[1]

    @cute.kernel
    def selection_offsets_kernel(
        self,
        mGreaterCounts: cute.Tensor,
        mEqualCounts: cute.Tensor,
        mGreaterOffsets: cute.Tensor,
        mEqualOffsets: cute.Tensor,
        mState: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        warp_offsets = smem.allocate_tensor(
            Int32,
            cute.make_layout((2, _WARPS)),
            byte_alignment=16,
        )
        totals = smem.allocate_tensor(
            Int32,
            cute.make_layout((2,)),
            byte_alignment=8,
        )
        blocks = mGreaterCounts.shape[0]
        begin = tidx * _OFFSET_ITEMS

        greater_thread = Int32(0)
        for item in cutlass.range_constexpr(_OFFSET_ITEMS):
            block = begin + item
            if block < blocks:
                greater_thread += Int32(mGreaterCounts[block])
        greater_prefix = _block_exclusive_sum(greater_thread, warp_offsets[0, None])
        running = greater_prefix
        for item in cutlass.range_constexpr(_OFFSET_ITEMS):
            block = begin + item
            if block < blocks:
                mGreaterOffsets[block] = running
                running += Int32(mGreaterCounts[block])
        if tidx == _THREADS - 1:
            totals[0] = greater_prefix + greater_thread

        equal_thread = Int32(0)
        for item in cutlass.range_constexpr(_OFFSET_ITEMS):
            block = begin + item
            if block < blocks:
                equal_thread += Int32(mEqualCounts[block])
        equal_prefix = _block_exclusive_sum(equal_thread, warp_offsets[1, None])
        running = equal_prefix
        for item in cutlass.range_constexpr(_OFFSET_ITEMS):
            block = begin + item
            if block < blocks:
                mEqualOffsets[block] = running
                running += Int32(mEqualCounts[block])
        if tidx == _THREADS - 1:
            totals[1] = equal_prefix + equal_thread
        cute.arch.sync_threads()

        if tidx == 0:
            greater_total = Int32(0)
            equal_total = Int32(0)
            if blocks <= _THREADS * _OFFSET_ITEMS:
                greater_total = Int32(totals[0])
                equal_total = Int32(totals[1])
            else:
                for block in cutlass.range(blocks, unroll=1):
                    mGreaterOffsets[block] = greater_total
                    mEqualOffsets[block] = equal_total
                    greater_total += Int32(mGreaterCounts[block])
                    equal_total += Int32(mEqualCounts[block])
            equal_needed = Int32(self.capacity) - greater_total
            if equal_needed < 0:
                equal_needed = Int32(0)
            if equal_needed > equal_total:
                equal_needed = equal_total
            if (mState[_NO_DROP] != 0) | (mState[_APPROXIMATE] != 0):
                equal_needed = Int32(0)
            mState[_TOTAL_GREATER] = greater_total
            mState[_EQUAL_NEEDED] = equal_needed

    @cute.kernel
    def selection_scatter_kernel(
        self,
        mProbabilities: cute.Tensor,
        mExperts: cute.Tensor,
        mGreaterOffsets: cute.Tensor,
        mEqualOffsets: cute.Tensor,
        mState: cute.Tensor,
        mSelectedIndices: cute.Tensor,
        mSelectedValid: cute.Tensor,
        mSelectedExperts: cute.Tensor,
        mLocalPositions: cute.Tensor,
        mTokenIds: cute.Tensor,
        mTopkSlots: cute.Tensor,
        mExpertCounts: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        warp_offsets = smem.allocate_tensor(
            Int32,
            cute.make_layout((2, _WARPS)),
            byte_alignment=16,
        )
        block_expert_counts = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.local_experts,)),
            byte_alignment=16,
        )
        block_expert_bases = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.local_experts,)),
            byte_alignment=16,
        )
        block_expert_ranks = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.local_experts,)),
            byte_alignment=16,
        )
        threshold = Int32(mState[_THRESHOLD])
        no_drop = Boolean(mState[_NO_DROP] != 0)
        num_elements = mProbabilities.shape[0] * self.columns
        indices = [Int32(0) for _ in range(_ITEMS)]
        experts = [Int32(0) for _ in range(_ITEMS)]
        greater_flags = [Boolean(False) for _ in range(_ITEMS)]
        equal_flags = [Boolean(False) for _ in range(_ITEMS)]

        for item in cutlass.range_constexpr(_ITEMS):
            index = Int32(bidx * _CHUNK + tidx * _ITEMS + item)
            indices[item] = index
            expert = Int32(0)
            key = Int32(0)
            valid = Boolean(False)
            if index < num_elements:
                row = index // self.columns
                column = index - row * self.columns
                expert = Int32(mExperts[row, column])
                probability = Float32(mProbabilities[row, column])
                key = llvm.bitcast(T.i32(), probability)
                valid = Boolean(
                    (expert >= self.local_start)
                    & (expert < self.local_end)
                    & (probability > 0.0)
                    & (key >= _MIN_NORMAL_KEY)
                )
            experts[item] = expert
            greater = Boolean(valid & (no_drop | (key > threshold)))
            equal = Boolean(valid & (not no_drop) & (key == threshold))
            greater_flags[item] = greater
            equal_flags[item] = equal

        greater_count = Int32(0)
        equal_count = Int32(0)
        for item in cutlass.range_constexpr(_ITEMS):
            greater_count += Int32(greater_flags[item])
            equal_count += Int32(equal_flags[item])
        greater_prefix = _block_exclusive_sum(greater_count, warp_offsets[0, None])
        equal_prefix = _block_exclusive_sum(equal_count, warp_offsets[1, None])

        if tidx < self.local_experts:
            block_expert_counts[tidx] = Int32(0)
            block_expert_ranks[tidx] = Int32(0)
        cute.arch.sync_threads()

        greater_block_offset = Int32(mGreaterOffsets[bidx])
        equal_block_offset = Int32(mEqualOffsets[bidx])
        total_greater = Int32(mState[_TOTAL_GREATER])
        equal_needed = Int32(mState[_EQUAL_NEEDED])
        local_greater = Int32(0)
        local_equal = Int32(0)
        outputs = [Int32(-1) for _ in range(_ITEMS)]
        local_experts = [Int32(0) for _ in range(_ITEMS)]
        for item in cutlass.range_constexpr(_ITEMS):
            output = Int32(-1)
            if greater_flags[item]:
                output = greater_block_offset + greater_prefix + local_greater
                local_greater += 1
            elif equal_flags[item]:
                equal_rank = equal_block_offset + equal_prefix + local_equal
                local_equal += 1
                if equal_rank < equal_needed:
                    output = total_greater + equal_rank
            if output >= self.capacity:
                output = Int32(-1)
            outputs[item] = output
            local_expert = experts[item] - self.local_start
            local_experts[item] = local_expert
            if output >= 0:
                cute.arch.atomic_add(
                    block_expert_counts.iterator + local_expert,
                    Int32(1),
                    scope="cta",
                )

        cute.arch.sync_threads()
        if tidx < self.local_experts:
            count = Int32(block_expert_counts[tidx])
            base = Int32(0)
            if count != 0:
                base = cute.arch.atomic_add(
                    mExpertCounts.iterator + tidx,
                    count,
                    scope="gpu",
                )
            block_expert_bases[tidx] = base
        cute.arch.sync_threads()

        for item in cutlass.range_constexpr(_ITEMS):
            output = outputs[item]
            if output >= 0 and output < self.capacity:
                index = indices[item]
                row = index // self.columns
                column = index - row * self.columns
                local_expert = local_experts[item]
                local_rank = cute.arch.atomic_add(
                    block_expert_ranks.iterator + local_expert,
                    Int32(1),
                    scope="cta",
                )
                local_position = Int32(block_expert_bases[local_expert]) + local_rank
                mSelectedIndices[output] = Int64(index)
                mSelectedValid[output] = Boolean(True)
                mSelectedExperts[output] = local_expert
                mLocalPositions[output] = local_position
                mTokenIds[output] = Int64(row)
                mTopkSlots[output] = Int64(column)

    @cute.kernel
    def finalize_counts_offsets_kernel(
        self,
        mExpertCounts: cute.Tensor,
        mAlignedCounts: cute.Tensor,
        mExpertOffsets: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        warp_offsets = smem.allocate_tensor(
            Int32,
            cute.make_layout((_WARPS,)),
            byte_alignment=16,
        )
        aligned = Int32(0)
        if tidx < self.local_experts:
            count = Int32(mExpertCounts[tidx])
            aligned = ((count + 127) // 128) * 128
            if aligned < 128:
                aligned = Int32(128)
            mAlignedCounts[tidx] = aligned
        offset = _block_exclusive_sum(aligned, warp_offsets)
        if tidx < self.local_experts:
            mExpertOffsets[tidx] = offset
        if tidx == self.local_experts - 1:
            mExpertOffsets[self.local_experts] = offset + aligned

    @cute.kernel
    def finalize_mapping_scatter_kernel(
        self,
        mSelectedValid: cute.Tensor,
        mSelectedExperts: cute.Tensor,
        mLocalPositions: cute.Tensor,
        mExpertOffsets: cute.Tensor,
        mPackedIdx: cute.Tensor,
        mPackedAssignment: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        assignment = Int32(bidx * _THREADS + tidx)
        if assignment < self.capacity:
            if mSelectedValid[assignment]:
                expert = Int32(mSelectedExperts[assignment])
                destination = Int32(mExpertOffsets[expert]) + Int32(mLocalPositions[assignment])
                mPackedIdx[assignment] = Int64(destination)
                if destination < self.max_packed_rows:
                    mPackedAssignment[destination] = assignment
            else:
                mPackedIdx[assignment] = Int64(0)


@jit_cache
def _compile_selective_radix(
    columns: int,
    local_start: int,
    local_experts: int,
    capacity: int,
    max_packed_rows: int,
):
    rows = cute.sym_int()
    blocks = cute.sym_int()
    probabilities = fake_tensor(Float32, (rows, columns), leading_dim=1)
    experts = fake_tensor(Uint16, (rows, columns), leading_dim=1)
    histogram = fake_tensor(Int32, (_CUSTOM_BINS,))
    state = fake_tensor(Int32, (_STATE_SIZE,))
    keys_a = fake_tensor(Int32, (rows * columns,))
    keys_b = fake_tensor(Int32, (rows * columns,))
    greater_counts = fake_tensor(Int32, (blocks,))
    equal_counts = fake_tensor(Int32, (blocks,))
    greater_offsets = fake_tensor(Int32, (blocks,))
    equal_offsets = fake_tensor(Int32, (blocks,))
    selected_indices = fake_tensor(Int64, (capacity,))
    selected_valid = fake_tensor(Boolean, (capacity,), divisibility=8)
    selected_experts = fake_tensor(Int32, (capacity,))
    local_positions = fake_tensor(Int32, (capacity,))
    token_ids = fake_tensor(Int64, (capacity,))
    topk_slots = fake_tensor(Int64, (capacity,))
    expert_counts = fake_tensor(Int32, (local_experts,))
    aligned_counts = fake_tensor(Int32, (local_experts,))
    expert_offsets = fake_tensor(Int32, (local_experts + 1,))
    packed_idx = fake_tensor(Int64, (capacity,))
    packed_assignment = fake_tensor(Int32, (max_packed_rows,))
    return cute.compile(
        SelectiveRadix(
            columns,
            local_start,
            local_experts,
            capacity,
            max_packed_rows,
        ),
        probabilities,
        experts,
        histogram,
        state,
        keys_a,
        keys_b,
        greater_counts,
        equal_counts,
        greater_offsets,
        equal_offsets,
        selected_indices,
        selected_valid,
        selected_experts,
        local_positions,
        token_ids,
        topk_slots,
        expert_counts,
        aligned_counts,
        expert_offsets,
        packed_idx,
        packed_assignment,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cute_op(
    "quack::_selective_radix",
    mutates_args={
        "histogram",
        "state",
        "keys_a",
        "keys_b",
        "greater_counts",
        "equal_counts",
        "greater_offsets",
        "equal_offsets",
        "selected_indices",
        "selected_valid",
        "selected_experts",
        "local_positions",
        "token_ids",
        "topk_slots",
        "expert_counts",
        "aligned_counts",
        "expert_offsets",
        "packed_idx",
        "packed_assignment",
    },
    device_types="cuda",
)
def _selective_radix_out(
    full_probs: torch.Tensor,
    experts: torch.Tensor,
    histogram: torch.Tensor,
    state: torch.Tensor,
    keys_a: torch.Tensor,
    keys_b: torch.Tensor,
    greater_counts: torch.Tensor,
    equal_counts: torch.Tensor,
    greater_offsets: torch.Tensor,
    equal_offsets: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_valid: torch.Tensor,
    selected_experts: torch.Tensor,
    local_positions: torch.Tensor,
    token_ids: torch.Tensor,
    topk_slots: torch.Tensor,
    expert_counts: torch.Tensor,
    aligned_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    packed_idx: torch.Tensor,
    packed_assignment: torch.Tensor,
    local_start: int,
    local_experts: int,
    capacity: int,
    max_packed_rows: int,
) -> None:
    capability = get_device_capacity(full_probs)
    if capability != (9, 0):
        raise NotImplementedError(
            f"selective_radix requires SM90 (Hopper); got sm_{capability[0]}{capability[1]}"
        )
    compiled = _compile_selective_radix(
        full_probs.shape[1],
        local_start,
        local_experts,
        capacity,
        max_packed_rows,
    )
    compiled(
        full_probs,
        experts,
        histogram,
        state,
        keys_a,
        keys_b,
        greater_counts,
        equal_counts,
        greater_offsets,
        equal_offsets,
        selected_indices,
        selected_valid,
        selected_experts,
        local_positions,
        token_ids,
        topk_slots,
        expert_counts,
        aligned_counts,
        expert_offsets,
        packed_idx,
        packed_assignment,
    )


def selective_radix(
    full_probs: torch.Tensor,
    experts: torch.Tensor,
    local_start: int,
    local_experts: int,
    capacity: int,
    max_packed_rows: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Select the highest-probability local routes and build packed metadata.

    ``max_packed_rows`` is an allocation bound supplied by the caller. It must
    cover ``expert_offsets[-1]`` in normal use; destinations beyond the bound
    are intentionally omitted from ``packed_assignment`` without synchronizing
    the device to validate the final offset.

    This implementation is currently supported only on SM90 GPUs.
    """
    if not full_probs.is_cuda or not experts.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if full_probs.device != experts.device:
        raise ValueError("inputs must be on the same CUDA device")
    if full_probs.dtype != torch.float32 or experts.dtype != torch.uint16:
        raise ValueError("full_probs must be float32 and experts must be uint16")
    if full_probs.ndim != _MATRIX_NDIM or experts.shape != full_probs.shape:
        raise ValueError("inputs must be equally shaped matrices")
    if full_probs.stride(1) != 1 or experts.stride(1) != 1:
        raise ValueError("the last input dimension must be contiguous")
    if not 1 <= local_experts <= _MAX_LOCAL_EXPERTS:
        raise ValueError("local_experts must be in [1, 256]")
    n_elements = full_probs.numel()
    if n_elements > torch.iinfo(torch.int32).max:
        raise ValueError("too many assignments")
    if not 1 <= capacity <= n_elements:
        raise ValueError("capacity must be in [1, full_probs.numel()]")
    if max_packed_rows < 0:
        raise ValueError("max_packed_rows must be nonnegative")

    device = full_probs.device
    blocks = (n_elements + _CHUNK - 1) // _CHUNK
    workspace_words = 2 * n_elements + _CUSTOM_BINS + 4 * blocks + _STATE_SIZE
    workspace = torch.empty(workspace_words, dtype=torch.int32, device=device)
    cursor = 0
    keys_a = workspace[cursor : cursor + n_elements]
    cursor += n_elements
    keys_b = workspace[cursor : cursor + n_elements]
    cursor += n_elements
    histogram = workspace[cursor : cursor + _CUSTOM_BINS]
    cursor += _CUSTOM_BINS
    greater_counts = workspace[cursor : cursor + blocks]
    cursor += blocks
    equal_counts = workspace[cursor : cursor + blocks]
    cursor += blocks
    greater_offsets = workspace[cursor : cursor + blocks]
    cursor += blocks
    equal_offsets = workspace[cursor : cursor + blocks]
    cursor += blocks
    state = workspace[cursor : cursor + _STATE_SIZE]
    histogram.zero_()
    state.zero_()

    selected_indices = torch.empty(capacity, dtype=torch.int64, device=device)
    selected_valid = torch.empty(capacity, dtype=torch.bool, device=device)
    selected_experts = torch.empty(capacity, dtype=torch.int32, device=device)
    local_positions = torch.empty(capacity, dtype=torch.int32, device=device)
    token_ids = torch.empty(capacity, dtype=torch.int64, device=device)
    topk_slots = torch.empty(capacity, dtype=torch.int64, device=device)
    expert_counts = torch.empty(local_experts, dtype=torch.int32, device=device)
    aligned_counts = torch.empty_like(expert_counts)
    expert_offsets = torch.empty(local_experts + 1, dtype=torch.int32, device=device)
    packed_idx = torch.empty(capacity, dtype=torch.int64, device=device)
    packed_assignment = torch.full((max_packed_rows,), -1, dtype=torch.int32, device=device)

    _selective_radix_out(
        full_probs,
        experts,
        histogram,
        state,
        keys_a,
        keys_b,
        greater_counts,
        equal_counts,
        greater_offsets,
        equal_offsets,
        selected_indices,
        selected_valid,
        selected_experts,
        local_positions,
        token_ids,
        topk_slots,
        expert_counts,
        aligned_counts,
        expert_offsets,
        packed_idx,
        packed_assignment,
        local_start,
        local_experts,
        capacity,
        max_packed_rows,
    )
    return (
        selected_indices,
        selected_valid,
        selected_experts,
        local_positions,
        token_ids,
        topk_slots,
        expert_counts,
        aligned_counts,
        expert_offsets,
        packed_idx,
        packed_assignment,
    )


__all__ = ["selective_radix"]
