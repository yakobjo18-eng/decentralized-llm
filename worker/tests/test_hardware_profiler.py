from worker.hardware.profiler import collect


def test_collect_returns_real_values():
    snapshot = collect()

    assert snapshot.cpu_logical_cores > 0
    assert snapshot.hostname
    assert snapshot.python_version
    assert snapshot.os

    # On this project's Computer 1 there is no CUDA GPU (verified via
    # nvidia-smi absence in Phase 0). collect() must report that
    # truthfully rather than assume a GPU exists, and must leave
    # gpu_name/gpu_vram_bytes at their honest "none detected" values.
    assert isinstance(snapshot.has_cuda_gpu, bool)
    if not snapshot.has_cuda_gpu:
        assert snapshot.gpu_name == ""
        assert snapshot.gpu_vram_bytes == 0


def test_total_ram_is_plausible_for_a_modern_machine():
    snapshot = collect()
    # Not asserting an exact number (that would be a fabricated
    # expectation) -- only that psutil produced something in a
    # plausible range for a real machine, not a zero/garbage value.
    assert snapshot.total_ram_bytes > 512 * 1024 * 1024
