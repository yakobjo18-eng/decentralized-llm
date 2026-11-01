"""Phase 7 test classification, shared by the Python test suites.

Every collected test gets exactly one class; a test without one stops
the run at collection. Definitions (docs/DEPLOYMENT_LAN.md):

  REAL_SINGLE_MACHINE  real code in ONE OS process on the machine running
                       the tests (in-process gRPC over 127.0.0.1 included).
                       Synthetic inputs are fine when the conclusion is
                       not about hardware, machines or deployment.
  REAL_MULTI_PROCESS   separate OS processes (orchestrator.exe, worker
                       processes) talking over real sockets, ONE machine.
  REAL_MULTI_MACHINE   separate physical computers. No test is in this
                       class yet: Computers 2 and 3 are unavailable.
  SIMULATED            the conclusion rests on simulated hardware
                       profiles, monkeypatched hardware/OS sources,
                       synthetic worker records, or stand-in services.

Each suite's conftest.py supplies MODULES (module -> class), OVERRIDES
({"module::test_name": class}) and, optionally, PATCH_RULE_MODULES:
modules where a test that uses the `monkeypatch` fixture (or a fixture
listed in PATCH_FIXTURES) is SIMULATED and any other test is
REAL_SINGLE_MACHINE. Run one class with e.g. `-m real_multi_process`.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

CLASSES = ("REAL_SINGLE_MACHINE", "REAL_MULTI_PROCESS", "REAL_MULTI_MACHINE", "SIMULATED")
_DESCRIPTIONS = {
    "REAL_SINGLE_MACHINE": "real code, one process, one machine",
    "REAL_MULTI_PROCESS": "separate OS processes over real sockets, one machine",
    "REAL_MULTI_MACHINE": "separate physical computers",
    "SIMULATED": "simulated hardware / synthetic records / stand-in services",
}
_results = collections.Counter()


def register(config) -> None:
    for c in CLASSES:
        config.addinivalue_line("markers", f"{c.lower()}: Phase 7 test class {c} ({_DESCRIPTIONS[c]})")


def classify(items, modules, overrides, patch_rule_modules=(), patch_fixtures=("monkeypatch",)) -> None:
    missing = []
    for item in items:
        module = Path(str(item.fspath)).stem
        name = item.originalname or item.name
        cls = overrides.get(f"{module}::{name}")
        if cls is None and module in patch_rule_modules:
            cls = "SIMULATED" if any(f in item.fixturenames for f in patch_fixtures) else "REAL_SINGLE_MACHINE"
        if cls is None:
            cls = modules.get(module)
        if cls not in CLASSES:
            missing.append(item.nodeid)
            continue
        item.add_marker(getattr(pytest.mark, cls.lower()))
        item.user_properties.append(("test_classification", cls))
    if missing:
        raise pytest.UsageError("tests without a Phase 7 classification (add them to conftest.py MODULES/OVERRIDES):\n  "
                                + "\n  ".join(missing))


def record(report) -> None:
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        cls = dict(report.user_properties).get("test_classification", "?")
        _results[(cls, report.outcome)] += 1


def summarize(terminalreporter) -> None:
    if not _results:
        return
    terminalreporter.write_sep("-", "test classification (Phase 7)")
    for cls in CLASSES:
        counts = {o: n for (c, o), n in _results.items() if c == cls}
        total = sum(counts.values())
        detail = ", ".join(f"{n} {o}" for o, n in sorted(counts.items())) or "none"
        terminalreporter.write_line(f"{cls:<20} {total:>4}  ({detail})")
