"""
_log_softmax dispatch tests

Verifies that torch._log_softmax / F.log_softmax:
  - produces correct results on flagos device
  - C++ stub routes to cuda backend (only backend currently registered for
    _log_softmax; FlagGems does not export a C++ log_softmax kernel)
  - dispatch log confirms the actual backend used when the C++ stub is reached
    via FLAGOS_DISABLE_FLAGGEMS_PY=1

Usage:
    pytest tests/integration/ops/test_log_softmax_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_log_softmax_subprocess(extra_env: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "import torch.nn.functional as F; "
        "x = torch.randn(4,8,device='flagos:0'); "
        "F.log_softmax(x, dim=-1)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestLogSoftmaxCorrectness:
    """torch.log_softmax / F.log_softmax correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1, 10), (2, 8, 64)])
    def test_log_softmax_shape(self, shape):
        torch.manual_seed(0)
        x = torch.randn(*shape, device=DEVICE)
        out = F.log_softmax(x, dim=-1)
        assert out.shape == shape
        assert out.device.type == "flagos"

    def test_log_softmax_exp_sums_to_one(self):
        torch.manual_seed(1)
        x = torch.randn(32, 64, device=DEVICE)
        out = F.log_softmax(x, dim=-1)
        sums = out.exp().sum(dim=-1).cpu()
        torch.testing.assert_close(sums, torch.ones(32), rtol=1e-4, atol=1e-4)

    def test_log_softmax_matches_cpu(self):
        torch.manual_seed(2)
        x_cpu = torch.randn(16, 32)
        ref = F.log_softmax(x_cpu, dim=-1)
        x = x_cpu.to(DEVICE)
        out = F.log_softmax(x, dim=-1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_log_softmax_dim0(self):
        torch.manual_seed(3)
        x_cpu = torch.randn(8, 16)
        ref = F.log_softmax(x_cpu, dim=0)
        x = x_cpu.to(DEVICE)
        out = F.log_softmax(x, dim=0)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_log_softmax_half(self):
        torch.manual_seed(4)
        x_cpu = torch.randn(8, 16, dtype=torch.float16)
        ref = F.log_softmax(x_cpu, dim=-1)
        x = x_cpu.to(DEVICE)
        out = F.log_softmax(x, dim=-1)
        assert out.dtype == torch.float16
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)


class TestLogSoftmaxCStub:
    """Verify the C++ stub is reached when the Python composite path is disabled."""

    def test_cstub_correctness(self):
        # FLAGOS_DISABLE_FLAGGEMS_PY=1 skips the Python decomposition that
        # otherwise short-circuits _log_softmax, forcing dispatch through the
        # C++ stub (register.cc::WrapperLogSoftmax -> log_softmax_stub).
        env = os.environ.copy()
        env["FLAGOS_DISABLE_FLAGGEMS_PY"] = "1"
        code = (
            "import torch_fl, torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "x_cpu = torch.randn(8, 32)\n"
            "ref = F.log_softmax(x_cpu, dim=-1)\n"
            "out = F.log_softmax(x_cpu.to('flagos:0'), dim=-1).cpu()\n"
            "torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    def test_dispatch_log_cuda(self):
        result = _run_log_softmax_subprocess({
            "FLAGOS_DISABLE_FLAGGEMS_PY": "1",
            "FLAGOS_LOG_DISPATCH": "1",
            "FLAGOS_OP__log_softmax": "cuda",
        })
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] _log_softmax -> cuda" in result.stderr
