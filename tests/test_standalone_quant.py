import ast
from pathlib import Path

import pytest
import torch

from standalone.quant import blockwise_quant, quant_ref, triton_quant


def test_standalone_quant_has_no_quack_dependency():
    source = Path("standalone/quant.py").read_text()
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        alias.name == "quack" or alias.name.startswith("quack.")
        for node in imports
        for alias in node.names
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_standalone_quant_matches_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(0)
    x = torch.randn(5, 256, device="cuda", dtype=dtype)
    actual, actual_scale = blockwise_quant(x)
    expected, expected_scale = quant_ref(x)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("blocks_per_program", [1, 2, 4, 8, 16])
def test_triton_quant_matches_reference(dtype, blocks_per_program):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(0)
    x = torch.randn(5, 256, device="cuda", dtype=dtype)
    # Ten quant blocks exercises the masked tail for grouping factors 4, 8, and 16.
    actual, actual_scale = triton_quant(x, blocks_per_program=blocks_per_program)
    expected, expected_scale = quant_ref(x)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
