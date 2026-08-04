"""Benchmark standalone CuTe and Triton blockwise FP8 quantization.

Each timed invocation is preceded by a device-side overwrite of a buffer larger
than three times L2. The flush is ordered on the same CUDA stream but occurs
before the start event, so its execution time is excluded from the result.
"""

import argparse
import statistics

import torch

from standalone.quant import blockwise_quant, quant_ref, triton_quant


def _cold_l2_bench(fn, flush: torch.Tensor, warmup: int, rep: int) -> list[float]:
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for start, end in zip(starts, ends):
        flush.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]


def _bandwidth_gbps(m: int, n: int, block_size: int, latency_us: float) -> float:
    # BF16 input + FP8 output + FP32 reciprocal scale output.
    transferred_bytes = m * n * (2 + 1) + m * (n // block_size) * 4
    return transferred_bytes / latency_us / 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--m", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--blocks-per-program", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    # Three full L2 turnovers, plus one byte to force ceil behavior.
    flush_bytes = 3 * props.L2_cache_size + 1
    flush = torch.empty(flush_bytes, device="cuda", dtype=torch.uint8)

    print(f"GPU: {props.name}")
    print(f"L2: {props.L2_cache_size / 2**20:.1f} MiB; flush: {flush_bytes / 2**20:.1f} MiB")
    print(
        f"N={args.n}, block_size={args.block_size}, "
        f"blocks_per_program={args.blocks_per_program}, dtype=bfloat16, reps={args.rep}"
    )
    print()
    print(
        f"{'M':>7}  {'implementation':>14}  {'median us':>10}  {'p20 us':>10}  {'p80 us':>10}  {'GB/s':>9}  {'speedup':>8}"
    )

    for m in args.m:
        torch.manual_seed(0)
        x = torch.randn(m, args.n, device="cuda", dtype=torch.bfloat16)

        # Compile both paths and verify values before timing.
        cute_out, cute_scale = blockwise_quant(x, args.block_size)
        triton_out, triton_scale = triton_quant(
            x, args.block_size, blocks_per_program=args.blocks_per_program
        )
        ref_out, ref_scale = quant_ref(x, args.block_size)
        torch.testing.assert_close(cute_out.float(), ref_out.float(), rtol=0, atol=0)
        torch.testing.assert_close(cute_scale, ref_scale, rtol=0, atol=0)
        torch.testing.assert_close(triton_out.float(), ref_out.float(), rtol=0, atol=0)
        torch.testing.assert_close(triton_scale, ref_scale, rtol=0, atol=0)

        timings = {}
        for name, fn in (
            ("CuTe", lambda: blockwise_quant(x, args.block_size)),
            (
                "Triton 1D",
                lambda: triton_quant(
                    x, args.block_size, blocks_per_program=args.blocks_per_program
                ),
            ),
        ):
            samples = _cold_l2_bench(fn, flush, args.warmup, args.rep)
            samples.sort()
            timings[name] = {
                "median": statistics.median(samples),
                "p20": samples[int(0.2 * (len(samples) - 1))],
                "p80": samples[int(0.8 * (len(samples) - 1))],
            }

        baseline = timings["Triton 1D"]["median"]
        for name in ("CuTe", "Triton 1D"):
            result = timings[name]
            speedup = baseline / result["median"]
            bandwidth = _bandwidth_gbps(m, args.n, args.block_size, result["median"])
            print(
                f"{m:7d}  {name:>14}  {result['median']:10.2f}  "
                f"{result['p20']:10.2f}  {result['p80']:10.2f}  "
                f"{bandwidth:9.1f}  {speedup:7.2f}x"
            )


if __name__ == "__main__":
    main()
