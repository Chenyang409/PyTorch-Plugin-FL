"""
_log_softmax_backward_data dispatch tests

Verifies that aten._log_softmax_backward_data.default:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_log_softmax_backward_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_log_softmax_backward_subprocess(
    extra_env: dict,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch, torch_fl; "
        "x = torch.randn(2, 4, device='flagos:0'); "
        "out = torch.log_softmax(x, dim=-1); "
        "grad = torch.randn_like(out); "
        "torch.ops.aten._log_softmax_backward_data.default(grad, out, -1, x.dtype)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestLogSoftmaxBackwardCorrectness:
    """_log_softmax_backward_data correctness on flagos device."""

    @pytest.mark.parametrize("shape, dim", [((2, 4), -1), ((3, 5), 1), ((2, 3, 4), -1)])
    def test_log_softmax_backward_matches_formula(self, shape, dim):
        torch.manual_seed(0)
        x_cpu = torch.randn(*shape)
        output_cpu = torch.log_softmax(x_cpu, dim=dim)
        grad_cpu = torch.randn_like(output_cpu)
        ref = grad_cpu - output_cpu.exp() * grad_cpu.sum(dim=dim, keepdim=True)

        output = output_cpu.to(DEVICE)
        grad = grad_cpu.to(DEVICE)
        out = torch.ops.aten._log_softmax_backward_data.default(
            grad, output, dim, x_cpu.dtype
        )

        assert out.shape == shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_log_softmax_backward_half_dtype(self):
        torch.manual_seed(1)
        x_cpu = torch.randn(4, 8, dtype=torch.float16)
        output_cpu = torch.log_softmax(x_cpu.float(), dim=-1).half()
        grad_cpu = torch.randn_like(output_cpu)

        output = output_cpu.to(DEVICE)
        grad = grad_cpu.to(DEVICE)
        out = torch.ops.aten._log_softmax_backward_data.default(
            grad, output, -1, torch.float16
        )

        assert out.shape == (4, 8)
        assert out.dtype == torch.float16
        assert out.device.type == "flagos"


class TestLogSoftmaxBackwardDispatch:
    """Verify C++ wrapper routes to the correct backend."""

    def test_dispatch_log_cuda(self):
        result = _run_log_softmax_backward_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP__log_softmax_backward_data": "cuda",
            }
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] _log_softmax_backward_data -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_log_softmax_backward_subprocess(
            {"FLAGOS_OP__log_softmax_backward_data": "flaggems"}
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
