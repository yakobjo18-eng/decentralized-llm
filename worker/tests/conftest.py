"""Phase 7: every worker test carries one test classification
(tests/classification.py at the repository root)."""

from __future__ import annotations

from classification import classify, record, register, summarize

MODULES = {
    "test_config_loader": "REAL_SINGLE_MACHINE",
    "test_config_phase3": "REAL_SINGLE_MACHINE",
    "test_execution_config": "REAL_SINGLE_MACHINE",
    "test_bind_policy": "REAL_SINGLE_MACHINE",
    "test_hardware_profiler": "REAL_SINGLE_MACHINE",
    "test_main_retry": "SIMULATED",          # fake registry client stands in for the orchestrator
    "test_registry_client": "SIMULATED",     # fake in-process WorkerRegistry server
    "test_simulated_profiles": "SIMULATED",
    "test_graceful_shutdown_integration": "REAL_MULTI_PROCESS",
    "test_three_worker_integration": "REAL_MULTI_PROCESS",
    "test_phase3_hardware_integration": "SIMULATED",
}
OVERRIDES = {
    # Real orchestrator + real worker processes, no simulated profile involved.
    "test_phase3_hardware_integration::test_worker_reregisters_with_fresh_profile_after_orchestrator_restart": "REAL_MULTI_PROCESS",
}
# Hardware detection modules: tests that monkeypatch detection sources
# simulate hardware this project lacks; the rest read Computer 1 itself.
PATCH_RULE_MODULES = ("test_hardware_capabilities", "test_hardware_schema3")
PATCH_FIXTURES = ("monkeypatch", "no_gpu_sources")


def pytest_configure(config):
    register(config)


def pytest_collection_modifyitems(config, items):
    classify(items, MODULES, OVERRIDES, PATCH_RULE_MODULES, PATCH_FIXTURES)


def pytest_runtest_logreport(report):
    record(report)


def pytest_terminal_summary(terminalreporter):
    summarize(terminalreporter)
