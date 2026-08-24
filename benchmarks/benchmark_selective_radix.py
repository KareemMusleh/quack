"""Benchmark the public selective-radix allocation path on SM90."""

import argparse
import gc
import math
import statistics
from dataclasses import dataclass

import torch

import cutlass

from quack.selective_radix import selective_radix


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    local_experts: int
    capacity: int
    distribution: str


CASES = (
    Case("no-drop", 4096, 64, 10240, "random"),
    Case("drop", 16384, 256, 40960, "random"),
    Case("all-tie", 16384, 256, 40960, "tie"),
)


def _inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(case.rows + case.local_experts)
    if case.distribution == "tie":
        scores = torch.ones((case.rows, 8), device="cuda", dtype=torch.float32)
        experts = torch.zeros((case.rows, 8), device="cuda", dtype=torch.uint16)
    else:
        scores = torch.rand((case.rows, 8), device="cuda", generator=generator)
        scores /= scores.sum(dim=1, keepdim=True)
        experts = torch.randint(
            256,
            (case.rows, 8),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        ).to(torch.uint16)
    return scores, experts


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _benchmark(case: Case, warmup: int, repetitions: int) -> tuple[float, float]:
    scores, experts = _inputs(case)
    max_packed_rows = math.ceil((case.capacity + case.local_experts * 128) / 128) * 128

    def run():
        return selective_radix(
            scores,
            experts,
            0,
            case.local_experts,
            case.capacity,
            max_packed_rows,
        )

    # Compile and lazy-load before the measured warm-up and timed calls.
    run()
    torch.cuda.synchronize()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for start, end in zip(starts, ends):
            start.record()
            run()
            end.record()
        torch.cuda.synchronize()
    finally:
        if was_enabled:
            gc.enable()

    latencies_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    return statistics.median(latencies_us), _percentile(latencies_us, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *(case.name for case in CASES)), default="all")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("benchmark_selective_radix requires an SM90 CUDA device")
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("--warmup must be nonnegative and --repetitions must be positive")

    cutlass.cuda.initialize_cuda_context()
    selected = CASES if args.case == "all" else tuple(c for c in CASES if c.name == args.case)
    print("case      rows assignments capacity median_us p95_us")
    for case in selected:
        median_us, p95_us = _benchmark(case, args.warmup, args.repetitions)
        print(
            f"{case.name:<9} {case.rows:>6} {case.rows * 8:>11} {case.capacity:>8} "
            f"{median_us:>9.3f} {p95_us:>7.3f}"
        )


if __name__ == "__main__":
    main()
