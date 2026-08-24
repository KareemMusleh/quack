import importlib
import math

import pytest
import torch

from quack import selective_radix
from quack.selective_radix import selective_radix as selective_radix_from_module


_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = pytest.mark.skipif(not _IS_SM90, reason="selective_radix requires SM90")

_OUTPUT_DTYPES = (
    torch.int64,
    torch.bool,
    torch.int32,
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int64,
    torch.int32,
)


def _max_packed_rows(capacity: int, local_experts: int) -> int:
    return math.ceil((capacity + local_experts * 128) / 128) * 128


def _custom_digit(keys: torch.Tensor) -> torch.Tensor:
    prefix5 = keys >> 27
    return torch.where(
        prefix5 <= 3,
        0,
        torch.where(
            prefix5 <= 5,
            1,
            torch.where(prefix5 == 6, 2, 3 + ((keys >> 19) & 0xFF)),
        ),
    )


def _reference_selected_indices(
    scores: torch.Tensor,
    experts: torch.Tensor,
    local_start: int,
    local_experts: int,
    capacity: int,
) -> torch.Tensor:
    flat_scores = scores.flatten()
    flat_experts = experts.flatten().to(torch.int32)
    eligible = (
        (flat_experts >= local_start)
        & (flat_experts < local_start + local_experts)
        & (flat_scores >= torch.finfo(torch.float32).smallest_normal)
    )
    indices = torch.arange(flat_scores.numel(), device=scores.device)[eligible]
    eligible_scores = flat_scores[eligible]
    if indices.numel() <= capacity:
        return indices

    keys = eligible_scores.view(torch.int32)
    digits = _custom_digit(keys)
    histogram = torch.bincount(digits.to(torch.int64), minlength=259).cpu().tolist()
    rank = capacity
    boundary = -1
    for digit in range(258, -1, -1):
        count = histogram[digit]
        if rank <= count:
            boundary = digit
            break
        rank -= count
    assert boundary >= 0

    if boundary < 3:
        threshold = (0x1FFFFFFF, 0x2FFFFFFF, 0x37FFFFFF)[boundary]
        return indices[keys > threshold]

    threshold = torch.topk(eligible_scores, capacity).values[-1]
    greater = indices[eligible_scores > threshold]
    equal = indices[eligible_scores == threshold]
    return torch.cat((greater, equal[: capacity - greater.numel()]))


def _assert_output(
    output: tuple[torch.Tensor, ...],
    scores: torch.Tensor,
    experts: torch.Tensor,
    local_start: int,
    local_experts: int,
    capacity: int,
    max_packed_rows: int,
) -> None:
    (
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
    ) = output

    assert len(output) == 11
    assert tuple(t.dtype for t in output) == _OUTPUT_DTYPES
    assert tuple(t.shape for t in output) == (
        (capacity,),
        (capacity,),
        (capacity,),
        (capacity,),
        (capacity,),
        (capacity,),
        (local_experts,),
        (local_experts,),
        (local_experts + 1,),
        (capacity,),
        (max_packed_rows,),
    )

    reference = _reference_selected_indices(scores, experts, local_start, local_experts, capacity)
    assert int(selected_valid.sum()) == reference.numel()
    torch.testing.assert_close(selected_indices[selected_valid], reference)
    assert not selected_valid[reference.numel() :].any()
    torch.testing.assert_close(
        selected_indices[~selected_valid], torch.zeros_like(selected_indices[~selected_valid])
    )
    torch.testing.assert_close(
        packed_idx[~selected_valid], torch.zeros_like(packed_idx[~selected_valid])
    )

    expected_experts = experts.flatten().to(torch.int32)[reference] - local_start
    torch.testing.assert_close(selected_experts[selected_valid], expected_experts)
    torch.testing.assert_close(token_ids[selected_valid], reference // scores.shape[1])
    torch.testing.assert_close(topk_slots[selected_valid], reference % scores.shape[1])

    expected_counts = torch.bincount(expected_experts.to(torch.int64), minlength=local_experts)
    expected_counts = expected_counts.to(torch.int32)
    torch.testing.assert_close(expert_counts, expected_counts)
    assert int(selected_valid.sum()) == int(expert_counts.sum())

    expected_aligned = torch.maximum(
        torch.full_like(expert_counts, 128),
        torch.div(expert_counts + 127, 128, rounding_mode="floor") * 128,
    )
    torch.testing.assert_close(aligned_counts, expected_aligned)
    torch.testing.assert_close(expert_offsets[0], torch.zeros_like(expert_offsets[0]))
    torch.testing.assert_close(expert_offsets[1:] - expert_offsets[:-1], aligned_counts)

    for expert in range(local_experts):
        mask = selected_valid & (selected_experts == expert)
        positions = local_positions[mask].sort().values
        torch.testing.assert_close(
            positions,
            torch.arange(positions.numel(), dtype=torch.int32, device=scores.device),
        )

    assignments = torch.arange(capacity, dtype=torch.int32, device=scores.device)
    destinations = packed_idx[selected_valid].long()
    torch.testing.assert_close(packed_assignment[destinations], assignments[selected_valid])
    written = torch.zeros(max_packed_rows, dtype=torch.bool, device=scores.device)
    written[destinations] = True
    torch.testing.assert_close(
        packed_assignment[~written], torch.full_like(packed_assignment[~written], -1)
    )


def _make_random_case(rows: int, local_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(rows + local_experts)
    scores = torch.rand((rows, 8), device="cuda", generator=generator)
    scores /= scores.sum(dim=1, keepdim=True)
    experts = torch.randint(
        256,
        (rows, 8),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    ).to(torch.uint16)
    return scores, experts


@pytest.mark.parametrize(
    ("rows", "local_experts", "capacity"),
    [
        (257, 8, 2056),
        (1024, 64, 512),
        (1024, 256, 2048),
        (65536, 64, 163840),
        pytest.param(131072, 8, 2056, id="serial-offset-fallback"),
    ],
)
def test_selective_radix_random_cases(
    rows: int,
    local_experts: int,
    capacity: int,
) -> None:
    scores, experts = _make_random_case(rows, local_experts)
    max_packed_rows = _max_packed_rows(capacity, local_experts)
    output = selective_radix(scores, experts, 0, local_experts, capacity, max_packed_rows)
    _assert_output(output, scores, experts, 0, local_experts, capacity, max_packed_rows)


def test_selective_radix_stable_threshold_ties_and_torch_compile() -> None:
    scores = torch.ones((128, 8), device="cuda", dtype=torch.float32)
    experts = torch.full((128, 8), 64, device="cuda", dtype=torch.uint16)
    capacity = 511
    local_start = 64
    local_experts = 64
    max_packed_rows = _max_packed_rows(capacity, local_experts)

    eager = selective_radix(scores, experts, local_start, local_experts, capacity, max_packed_rows)
    torch.testing.assert_close(eager[0], torch.arange(capacity, dtype=torch.int64, device="cuda"))
    _assert_output(
        eager,
        scores,
        experts,
        local_start,
        local_experts,
        capacity,
        max_packed_rows,
    )

    def fn(full_probs: torch.Tensor, expert_ids: torch.Tensor):
        return selective_radix(
            full_probs,
            expert_ids,
            local_start,
            local_experts,
            capacity,
            max_packed_rows,
        )

    compiled = torch.compile(fn, fullgraph=True)(scores, experts)
    _assert_output(
        compiled,
        scores,
        experts,
        local_start,
        local_experts,
        capacity,
        max_packed_rows,
    )


def test_selective_radix_strides_special_values_and_cardinality_edges() -> None:
    rows = 8
    columns = 8
    local_start = 17
    local_experts = 1
    capacity = 8
    max_packed_rows = 256
    score_storage = torch.zeros((rows, columns + 2), device="cuda")
    expert_storage = torch.zeros((rows, columns + 2), dtype=torch.uint16, device="cuda")
    scores = score_storage[:, :columns]
    experts = expert_storage[:, :columns]
    assert scores.stride() == (columns + 2, 1)
    assert experts.stride() == (columns + 2, 1)

    scores[0] = torch.tensor(
        [
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            torch.finfo(torch.float32).smallest_normal / 2,
            0.5,
            0.25,
            float("-inf"),
        ],
        dtype=torch.float32,
        device="cuda",
    )
    experts[0] = local_start
    output = selective_radix(scores, experts, local_start, local_experts, capacity, max_packed_rows)
    _assert_output(output, scores, experts, local_start, local_experts, capacity, max_packed_rows)

    scores.fill_(1.0)
    experts.zero_()
    output = selective_radix(scores, experts, local_start, local_experts, capacity, max_packed_rows)
    _assert_output(output, scores, experts, local_start, local_experts, capacity, max_packed_rows)

    scores.zero_()
    experts.zero_()
    scores[0] = 1.0
    experts[0] = local_start
    output = selective_radix(scores, experts, local_start, local_experts, capacity, max_packed_rows)
    _assert_output(output, scores, experts, local_start, local_experts, capacity, max_packed_rows)


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_selective_radix_preserves_approximate_low_buckets(boundary: int) -> None:
    scores = torch.zeros((8, 8), dtype=torch.float32, device="cuda")
    experts = torch.zeros((8, 8), dtype=torch.uint16, device="cuda")
    boundary_key = (0x10000000, 0x20000000, 0x30000000)[boundary]
    higher_key = (0x20000000, 0x30000000, 0x38000000)[boundary]
    keys = torch.tensor([higher_key] * 4 + [boundary_key] * 8, dtype=torch.int32, device="cuda")
    scores.flatten()[: keys.numel()].copy_(keys.view(torch.float32))
    capacity = 8
    max_packed_rows = 256

    output = selective_radix(scores, experts, 0, 1, capacity, max_packed_rows)
    assert int(output[1].sum()) == 4
    _assert_output(output, scores, experts, 0, 1, capacity, max_packed_rows)


def test_selective_radix_public_imports_match() -> None:
    assert selective_radix is selective_radix_from_module


def test_selective_radix_omits_reverse_mapping_beyond_allocation_bound() -> None:
    scores = torch.ones((2, 8), device="cuda", dtype=torch.float32)
    experts = torch.ones((2, 8), device="cuda", dtype=torch.uint16)
    output = selective_radix(scores, experts, 0, 2, 16, 128)

    assert output[1].all()
    assert (output[9] >= 128).all()
    torch.testing.assert_close(output[10], torch.full_like(output[10], -1))


@pytest.mark.parametrize("local_experts", [0, 257])
def test_selective_radix_rejects_invalid_local_expert_count(local_experts: int) -> None:
    scores = torch.ones((2, 8), device="cuda", dtype=torch.float32)
    experts = torch.zeros((2, 8), device="cuda", dtype=torch.uint16)
    with pytest.raises(ValueError, match="local_experts"):
        selective_radix(scores, experts, 0, local_experts, 1, 128)


@pytest.mark.parametrize("capacity", [0, 17])
def test_selective_radix_rejects_invalid_capacity(capacity: int) -> None:
    scores = torch.ones((2, 8), device="cuda", dtype=torch.float32)
    experts = torch.zeros((2, 8), device="cuda", dtype=torch.uint16)
    with pytest.raises(ValueError, match="capacity"):
        selective_radix(scores, experts, 0, 1, capacity, 128)


def test_selective_radix_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = torch.ones((2, 8), device="cuda", dtype=torch.float32)
    experts = torch.zeros((2, 8), device="cuda", dtype=torch.uint16)

    with pytest.raises(ValueError, match="float32.*uint16"):
        selective_radix(scores.to(torch.float16), experts, 0, 1, 1, 128)
    with pytest.raises(ValueError, match="float32.*uint16"):
        selective_radix(scores, experts.to(torch.int32), 0, 1, 1, 128)
    with pytest.raises(ValueError, match="equally shaped matrices"):
        selective_radix(scores[0], experts[0], 0, 1, 1, 128)
    with pytest.raises(ValueError, match="equally shaped matrices"):
        selective_radix(scores, experts[:, :7], 0, 1, 1, 128)
    with pytest.raises(ValueError, match="last input dimension"):
        selective_radix(scores[:, ::2], experts[:, ::2], 0, 1, 1, 128)
    with pytest.raises(ValueError, match="max_packed_rows"):
        selective_radix(scores, experts, 0, 1, 1, -1)
    with pytest.raises(ValueError, match="CUDA tensors"):
        selective_radix(scores.cpu(), experts.cpu(), 0, 1, 1, 128)

    module = importlib.import_module("quack.selective_radix")
    monkeypatch.setattr(module, "get_device_capacity", lambda _tensor: (10, 0))
    with pytest.raises(NotImplementedError, match="requires SM90"):
        selective_radix(scores, experts, 0, 1, 1, 128)


def test_selective_radix_rejects_different_devices() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    scores = torch.ones((2, 8), device="cuda:0", dtype=torch.float32)
    experts = torch.zeros((2, 8), device="cuda:1", dtype=torch.uint16)
    with pytest.raises(ValueError, match="same CUDA device"):
        selective_radix(scores, experts, 0, 1, 1, 128)
