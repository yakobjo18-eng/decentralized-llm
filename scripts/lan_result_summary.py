"""Print the record a physical LAN test needs, from an execution result.

    python scripts/lan_result_summary.py lan-result.json [--label "first run"]

Reads the JSON written by `python -m partitioner execute ... --remote
--out lan-result.json` and prints, as a Markdown table row set: host
identities, worker ids, model and plan ids, output shape, the reference
comparison and its tolerance, elapsed time, and the test classification.
Host identities are what the workers reported about themselves
(unauthenticated); the classification is derived from them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    r = json.loads(Path(args.result).read_text(encoding="utf-8"))
    ref = r.get("reference_check") or {}
    rows = [
        ("label", args.label or "-"),
        ("status", r.get("status")),
        ("test_classification", r.get("test_classification") or "none (no stage ran)"),
        ("classification basis", r.get("test_classification_basis") or "-"),
        ("coordinator", f"{r['coordinator']['hostname']} (pid {r['coordinator']['pid']})" if r.get("coordinator") else "-"),
        ("model_id", (r.get("model") or {}).get("model_id")),
        ("plan_id", (r.get("plan") or {}).get("plan_id")),
        ("request_id", r.get("request_id")),
        ("output shape", (r.get("output") or {}).get("shape")),
        ("next_token_ids", (r.get("output") or {}).get("next_token_ids")),
        ("reference max abs diff", ref.get("max_abs_diff")),
        ("tolerance (atol)", ref.get("atol")),
        ("reference passed", ref.get("passed")),
        ("elapsed (coordinator wall clock, single run)", f"{r['elapsed']['total_seconds']:.3f} s" if r.get("elapsed") else "-"),
    ]
    for s in r.get("stages", []):
        rep = s.get("worker_reported") or {}
        rows.append((f"stage {s['stage_index']}",
                     f"worker {s['worker_id']} @ {s['endpoint']} -- reported host {rep.get('hostname')!r} pid {rep.get('pid')}, "
                     f"units {s['unit_range']}, round trip {s.get('round_trip_seconds', 0):.4f} s, "
                     f"attempts {[a['outcome'] for a in s.get('attempts', [])]}"))
    if r.get("error"):
        e = r["error"]
        rows.append(("error", f"{e.get('phase')} / {e.get('code') or e.get('type')} at stage {e.get('stage_index')}: "
                              f"{(e.get('message') or '').splitlines()[0][:200]}"))
    print("| Field | Value |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {str(v).replace('|', '/')} |")
    return 0 if r.get("status") == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
