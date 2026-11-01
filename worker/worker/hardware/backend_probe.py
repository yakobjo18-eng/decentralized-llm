"""Inference backend capability probing (separate from GPU hardware
detection -- a GPU being present says nothing about whether a framework
can compute on it, and vice versa).

For each backend the installed PyTorch build knows about, this records
whether the framework reports it available, and -- if so and if
run_compute_test is set -- runs a tiny compute test on it: a 256x256
float32 matmul whose result is checked against a known answer (A @ I
must equal A), then timed over a few repetitions for a rough GFLOPS
figure. No model is downloaded, nothing is installed; the whole test
allocates well under 1 MB per device.

Backends probed: cpu, cuda, xpu (Intel GPUs, needs an XPU build of
torch), mps (Apple), directml (the separate torch-directml package).
Unknown future backends can be added to _PROBES without any schema
change -- BackendCapability.name is an open set.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, List, Optional

_TEST_SIZE = 256
_TEST_REPEATS = 8


@dataclasses.dataclass
class BackendCapability:
    name: str
    # None = could not be determined (never probed), distinct from False.
    available: Optional[bool]
    device_count: int = 0
    version: str = ""
    devices: List[str] = dataclasses.field(default_factory=list)
    tested: bool = False
    test_passed: Optional[bool] = None
    test_gflops: Optional[float] = None
    detail: str = ""


def run_matmul_test(torch, device) -> BackendCapability:
    """Runs the compute test on `device` and returns a partially-filled
    BackendCapability (tested/test_passed/test_gflops/detail only).
    Never raises."""
    result = BackendCapability(name="", available=True, tested=True)
    try:
        with torch.no_grad():
            gen = torch.Generator().manual_seed(1234)
            a_cpu = torch.rand((_TEST_SIZE, _TEST_SIZE), generator=gen, dtype=torch.float32)
            a = a_cpu.to(device)
            eye = torch.eye(_TEST_SIZE, dtype=torch.float32).to(device)
            product = (a @ eye).to("cpu")
            if not torch.allclose(product, a_cpu, atol=1e-5):
                result.test_passed = False
                result.detail = "matmul result did not match expected values"
                return result

            b = a @ a  # warm-up (kernel selection, lazy init)
            _sync(torch, device)
            start = time.perf_counter()
            for _ in range(_TEST_REPEATS):
                b = a @ a
            _sync(torch, device)
            _ = b.to("cpu")
            elapsed = time.perf_counter() - start
        result.test_passed = True
        if elapsed > 0:
            flops = 2.0 * (_TEST_SIZE**3) * _TEST_REPEATS
            result.test_gflops = round(flops / elapsed / 1e9, 2)
        result.detail = f"{_TEST_SIZE}x{_TEST_SIZE} float32 matmul verified"
    except Exception as exc:  # a failed test is a result, not a crash
        result.test_passed = False
        result.detail = f"compute test failed: {type(exc).__name__}: {exc}"[:300]
    return result


def _sync(torch, device) -> None:
    kind = getattr(device, "type", str(device)).split(":")[0]
    try:
        if kind == "cuda":
            torch.cuda.synchronize()
        elif kind == "xpu":
            torch.xpu.synchronize()
        elif kind == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def _merge_test(cap: BackendCapability, test: BackendCapability) -> BackendCapability:
    cap.tested = test.tested
    cap.test_passed = test.test_passed
    cap.test_gflops = test.test_gflops
    cap.detail = test.detail
    return cap


def _probe_cpu(torch, run_test: bool) -> BackendCapability:
    cap = BackendCapability(name="cpu", available=True, device_count=1)
    try:
        cap.devices = [f"cpu ({torch.get_num_threads()} torch threads)"]
    except Exception:
        pass
    if run_test:
        _merge_test(cap, run_matmul_test(torch, torch.device("cpu")))
    return cap


def _probe_cuda(torch, run_test: bool) -> BackendCapability:
    build = str(getattr(getattr(torch, "version", None), "cuda", None) or "")
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return BackendCapability(name="cuda", available=None, version=build, detail=f"probe failed: {exc}"[:300])
    cap = BackendCapability(name="cuda", available=available, version=build)
    if not available:
        cap.detail = (
            "torch build has no CUDA support"
            if not build
            else f"torch built for CUDA {build}, but torch.cuda.is_available() is False (no usable NVIDIA GPU/driver)"
        )
        return cap
    try:
        cap.device_count = int(torch.cuda.device_count())
        cap.devices = [str(torch.cuda.get_device_name(i)) for i in range(cap.device_count)]
    except Exception:
        pass
    if run_test:
        _merge_test(cap, run_matmul_test(torch, torch.device("cuda:0")))
    return cap


def _probe_xpu(torch, run_test: bool) -> Optional[BackendCapability]:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return None  # this torch build doesn't know the backend at all
    try:
        available = bool(xpu.is_available())
    except Exception as exc:
        return BackendCapability(name="xpu", available=None, detail=f"probe failed: {exc}"[:300])
    cap = BackendCapability(name="xpu", available=available)
    if not available:
        cap.detail = "torch.xpu.is_available() is False (no Intel GPU runtime in this torch build)"
        return cap
    try:
        cap.device_count = int(xpu.device_count())
        cap.devices = [str(xpu.get_device_name(i)) for i in range(cap.device_count)]
    except Exception:
        pass
    if run_test:
        _merge_test(cap, run_matmul_test(torch, torch.device("xpu:0")))
    return cap


def _probe_mps(torch, run_test: bool) -> Optional[BackendCapability]:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None:
        return None
    try:
        available = bool(mps.is_available())
    except Exception as exc:
        return BackendCapability(name="mps", available=None, detail=f"probe failed: {exc}"[:300])
    cap = BackendCapability(name="mps", available=available, device_count=1 if available else 0)
    if not available:
        cap.detail = "Apple Metal backend not available (not an Apple Silicon Mac)"
        return cap
    if run_test:
        _merge_test(cap, run_matmul_test(torch, torch.device("mps")))
    return cap


def _probe_directml(torch, run_test: bool) -> BackendCapability:
    try:
        import torch_directml  # optional separate package; never installed by this code
    except ImportError:
        return BackendCapability(name="directml", available=False, detail="torch-directml package not installed")
    except Exception as exc:
        return BackendCapability(name="directml", available=None, detail=f"import failed: {exc}"[:300])
    cap = BackendCapability(name="directml", available=True, version=str(getattr(torch_directml, "__version__", "")))
    try:
        cap.device_count = int(torch_directml.device_count())
        cap.devices = [str(torch_directml.device_name(i)) for i in range(cap.device_count)]
    except Exception:
        pass
    if run_test:
        _merge_test(cap, run_matmul_test(torch, torch_directml.device()))
    return cap


_PROBES: List[Callable] = [_probe_cpu, _probe_cuda, _probe_xpu, _probe_mps, _probe_directml]


def probe_backends(torch, run_compute_test: bool = True) -> List[BackendCapability]:
    """Probe every known backend against an already-imported torch
    module. Each probe is isolated: one backend's failure never hides
    another's result."""
    out: List[BackendCapability] = []
    for probe in _PROBES:
        try:
            cap = probe(torch, run_compute_test)
        except Exception as exc:
            name = probe.__name__.replace("_probe_", "")
            cap = BackendCapability(name=name, available=None, detail=f"probe crashed: {exc}"[:300])
        if cap is not None:
            out.append(cap)
    return out


# ---------------------------------------------------------------------
# Subprocess entry point
#
# `import torch` takes 8-14 s on Computer 1 and ~500 MB of memory. The
# worker therefore runs the probe in a child process
# (`python -m worker.hardware.backend_probe`): registration doesn't
# wait for it, the worker never keeps torch loaded just to describe
# itself, and a GPU driver crashing during the probe kills only the
# child.
# ---------------------------------------------------------------------

REPORT_PREFIX = "BACKEND_PROBE_JSON:"


def probe_report(run_compute_test: bool = True) -> dict:
    try:
        import torch
    except Exception as exc:  # ImportError, or a broken install
        return {"torch_installed": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "torch_installed": True,
        "torch_version": str(getattr(torch, "__version__", "") or ""),
        "torch_cuda_build_version": str(getattr(getattr(torch, "version", None), "cuda", None) or ""),
        "capabilities": [dataclasses.asdict(c) for c in probe_backends(torch, run_compute_test)],
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe inference backends and print a JSON report")
    parser.add_argument("--no-compute-test", action="store_true")
    args = parser.parse_args(argv)
    # Prefixed so stray library output on stdout can't be mistaken for the report.
    print(REPORT_PREFIX + json.dumps(probe_report(run_compute_test=not args.no_compute_test)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
