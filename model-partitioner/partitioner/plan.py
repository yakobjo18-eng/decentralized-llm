"""Partition plan schema (version 1) helpers. The full schema is in
docs/PARTITION_PLAN_SCHEMA.md; plans are plain JSON-serializable dicts
so any later phase (or language) can read them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

PLAN_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1,)
PLANNER_NAME = "model-partitioner"
PLANNER_VERSION = "0.1.0"
ALGORITHM = "dp-min-bottleneck-v1"


def fingerprint(plan: Dict[str, Any]) -> str:
    """sha256 of the plan's content excluding its timestamp and the
    fingerprint itself -- identical inputs give identical fingerprints."""
    content = {k: v for k, v in plan.items() if k not in ("created_at", "inputs_fingerprint")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def save(plan: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(plan, indent=2, sort_keys=False), encoding="utf-8")


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_bytes(n) -> str:
    if n is None:
        return "unknown"
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} B"
