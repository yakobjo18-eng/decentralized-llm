"""Phase 7 local operational validation (Computer 1, loopback).

    python scripts/phase7_local_validation.py --out docs/validation/phase7-local-<date>

Starts the REAL orchestrator.exe and three REAL `python -m worker.main`
processes (execution enabled, 127.0.0.1) and runs, in order:

  preflight (configs, live orchestrator, endpoints, plan) -> listener
  exposure audit -> planning from live registrations -> 2- and 3-worker
  remote pipelines -> worker resource-limit failure -> malformed
  requests -> timeout -> graceful worker stop/restart -> hard kill and
  restart -> orchestrator restart (and execution while it is down) ->
  clean shutdown with deregistration.

Every step is classified REAL_MULTI_PROCESS (separate processes, one
machine) unless stated otherwise; nothing here is REAL_MULTI_MACHINE.
Timings are single-run wall clock, NOT benchmarks. Writes report.json,
report.md and every process log into --out.

Optional --lan-ip IP additionally binds one stage-execution process to
this machine's private LAN IP. That exercises the LAN bind path, but
traffic to your own IP never leaves the machine. It may also make
Windows Firewall show a prompt, so it is off by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model-partitioner"), str(REPO / "worker")]

ORCH_EXE = REPO / "orchestrator" / "orchestrator.exe"
IDS2 = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
IDS1 = [[1, 2, 3, 4, 5, 6, 7, 8]]
READY = "StageExecution (protocol v1, UNAUTHENTICATED) listening on"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    """A child process whose combined output is streamed to a log file."""

    def __init__(self, name: str, args, cwd, log_dir: Path):
        self.name = name
        self.log_path = log_dir / f"{name}.log"
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.text = []
        self._log = self.log_path.open("a", encoding="utf-8")
        self._log.write(f"===== start {dt.datetime.now().isoformat()} : {' '.join(map(str, args))}\n")
        self.proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self.started = time.perf_counter()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self._log.write(line)
            self._log.flush()
            self.text.append(line)
            self.lines.put(line)

    def wait_for(self, marker: str, timeout: float) -> str:
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self.lines.get(timeout=0.3)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.name} exited with {self.proc.returncode}")
                continue
            if marker in line:
                return line
        raise TimeoutError(f"{self.name}: {marker!r} not seen within {timeout}s")

    def graceful_stop(self, timeout: float = 20) -> int:
        self.proc.send_signal(signal.CTRL_BREAK_EVENT)
        return self.proc.wait(timeout=timeout)

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def close(self):
        self.kill()
        time.sleep(0.2)
        self._log.close()


class Validation:
    def __init__(self, out: Path, lan_ip: str = ""):
        self.out = out
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.work = out / "work"
        self.work.mkdir(exist_ok=True)
        self.lan_ip = lan_ip
        self.steps = []
        self.procs = {}
        self.rest, self.grpc = free_port(), free_port()
        self.url = f"http://127.0.0.1:{self.rest}"
        self.workers = {
            "val-a": {"health": free_port(), "exec": free_port(), "extra": ""},
            "val-b": {"health": free_port(), "exec": free_port(), "extra": ""},
            # val-c accepts batch 1 only: used for the resource-limit failure.
            "val-c": {"health": free_port(), "exec": free_port(), "extra": "execution_max_batch: 1\n"},
        }
        self.endpoints = {w: f"127.0.0.1:{c['exec']}" for w, c in self.workers.items()}

    # ---- bookkeeping ---------------------------------------------------

    def step(self, sid, title, fn, classification="REAL_MULTI_PROCESS", expect_ok=True):
        t = time.perf_counter()
        print(f"\n--- {sid}: {title}", flush=True)
        try:
            ok, detail, evidence = fn()
        except Exception as exc:
            ok, detail, evidence = False, f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()}
        seconds = time.perf_counter() - t
        status = "PASS" if ok == expect_ok or ok is True else "FAIL"
        if not expect_ok:
            status = "PASS" if ok else "FAIL"
        self.steps.append({"id": sid, "title": title, "classification": classification, "status": status,
                           "detail": detail, "seconds": round(seconds, 3), "evidence": evidence})
        print(f"    {status} ({seconds:.2f}s): {detail}", flush=True)
        return status == "PASS"

    # ---- processes -----------------------------------------------------

    def orch_config(self) -> Path:
        p = self.work / "orchestrator.yaml"
        p.write_text(f'rest_address: "127.0.0.1:{self.rest}"\ngrpc_address: "127.0.0.1:{self.grpc}"\n'
                     'log_level: "info"\nheartbeat_timeout_seconds: 3\nallow_simulated_workers: false\n')
        return p

    def worker_config(self, wid) -> Path:
        c = self.workers[wid]
        p = self.work / f"{wid}.yaml"
        p.write_text(f'worker_id: "{wid}"\norchestrator_address: "127.0.0.1:{self.grpc}"\nlisten_host: "127.0.0.1"\n'
                     f'listen_port: {c["health"]}\nheartbeat_interval_seconds: 0.5\nbackend_compute_test: false\n'
                     f'execution_enabled: true\nexecution_listen_host: "127.0.0.1"\nexecution_listen_port: {c["exec"]}\n'
                     'log_level: "info"\n' + c["extra"])
        return p

    def start_orchestrator(self, label="orchestrator"):
        p = Proc(label, [str(ORCH_EXE), "--config", str(self.orch_config())], REPO / "orchestrator", self.logs)
        self.procs["orchestrator"] = p
        self.wait(lambda: self.get("/health")["status"] == "ok", 15, "orchestrator health")
        return p

    def start_worker(self, wid):
        p = Proc(wid, [sys.executable, "-m", "worker.main", "--config", str(self.worker_config(wid))], REPO / "worker", self.logs)
        self.procs[wid] = p
        p.wait_for(READY, 180)
        return p

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=3) as r:
            return json.loads(r.read().decode())

    def workers_view(self):
        return {w["id"]: w for w in self.get("/api/workers")}

    @staticmethod
    def wait(pred, timeout, what):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if pred():
                    return time.time() - (end - timeout)
            except Exception:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"timed out after {timeout}s waiting for {what}")

    # ---- helpers -------------------------------------------------------

    def plan(self, wids, batch, records):
        from partitioner import create_plan
        from partitioner.cost import CostModel
        from partitioner.execution.tiny_gpt2 import spec
        from partitioner.memory import Workload

        return create_plan(spec(), [records[w] for w in wids], workload=Workload(batch_size=batch),
                           cost_model=CostModel(throughput_preference="estimated"), allow_unused_workers=False)

    def run_remote(self, plan, ids, **opts):
        from partitioner.execution.remote import RemoteOptions, execute_plan_remote

        return execute_plan_remote(plan, ids, {w: self.endpoints[w] for w in {s["worker_id"] for s in plan["stages"]}},
                                   options=RemoteOptions(**{"connect_timeout_seconds": 3, **opts}))

    @staticmethod
    def summary(r):
        return {
            "status": r["status"], "test_classification": r.get("test_classification"),
            "distinct_worker_processes": r.get("distinct_worker_processes"),
            "distinct_worker_hostnames": r.get("distinct_worker_hostnames"),
            "plan_id": (r.get("plan") or {}).get("plan_id"), "model_id": (r.get("model") or {}).get("model_id"),
            "output_shape": (r.get("output") or {}).get("shape"), "next_token_ids": (r.get("output") or {}).get("next_token_ids"),
            "reference_check": r.get("reference_check"), "elapsed_seconds": r["elapsed"]["total_seconds"],
            "stages": [{"stage": s["stage_index"], "worker_id": s["worker_id"], "worker_reported": s.get("worker_reported"),
                        "units": s["unit_range"], "round_trip_seconds": s.get("round_trip_seconds"),
                        "worker_timing": s.get("worker_timing"), "attempts": [a["outcome"] for a in s.get("attempts", [])]}
                       for s in r["stages"]],
            "error": None if not r["error"] else {k: r["error"].get(k) for k in ("phase", "code", "type", "message",
                                                                                 "stage_index", "executed", "problems")}
                     | {"attempts": [a["outcome"] for a in r["error"].get("attempts", [])]},
        }

    def raw_request(self, wid, plan, stage, **change):
        """Hand-built ExecuteStage calls for the malformed-request step."""
        import grpc

        from partitioner.execution import wire
        from partitioner.execution.interface import INPUT_STAGE, STAGE_INTERFACE_VERSION, Activation, ActivationMeta
        from partitioner.execution.tiny_gpt2 import token_tensor
        from worker.generated import stage_execution_pb2_grpc as pb_grpc

        pb = wire.pb2()
        st = plan["stages"][stage]
        x = token_tensor(IDS2)
        payload = wire.encode(Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "m", "token_ids", tuple(x.shape), "int64",
                                                        INPUT_STAGE), x))
        fields = dict(protocol_version=pb.PROTOCOL_VERSION_1, request_id="malformed-test", plan_id=plan["inputs_fingerprint"],
                      plan_json=json.dumps(plan).encode(), model_id=plan["model"]["model_id"],
                      config_fingerprint=plan["model"]["config_fingerprint"], target_worker_id=wid, stage_index=stage,
                      unit_start=st["unit_range"][0], unit_end=st["unit_range"][1], input=payload)
        timeout = change.pop("_timeout", 10)
        mutate = change.pop("_mutate", None)
        fields.update(change)
        req = pb.ExecuteStageRequest(**fields)
        if mutate:
            mutate(req)
        with grpc.insecure_channel(self.endpoints[wid]) as ch:
            stub = pb_grpc.StageExecutionStub(ch)
            resp = stub.ExecuteStage(req, timeout=timeout) if timeout else stub.ExecuteStage(req)
        return {"status": pb.StageStatus.Name(resp.status), "code": resp.error.code, "message": resp.error.message,
                "executed": resp.error.executed}

    # ---- the run ---------------------------------------------------------

    def run(self):
        from partitioner import preflight
        from partitioner.workers import from_orchestrator

        meta = {"started_at": dt.datetime.now().isoformat(), "hostname": socket.gethostname(),
                "platform": platform.platform(), "python": sys.version.split()[0], "orchestrator_exe": str(ORCH_EXE),
                "rest": self.url, "grpc_port": self.grpc, "workers": self.workers, "endpoints": self.endpoints}
        state = {}

        def preflight_configs():
            r = preflight.run(worker_configs=[str(self.worker_config(w)) for w in self.workers], skip_runtime=False)
            return not r.failed, f"{r.counts()} (worker configs + runtime, before start)", {"report": r.to_dict()}

        self.step("S01", "preflight: runtime and the three worker configs", preflight_configs, "REAL_SINGLE_MACHINE")

        def start_all():
            t0 = time.perf_counter()
            self.start_orchestrator()
            t_orch = time.perf_counter() - t0
            starts = {}
            threads = []
            for wid in self.workers:  # start in parallel: each imports torch + transformers
                th = threading.Thread(target=lambda w=wid: starts.__setitem__(w, self._timed_start(w)))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
            waited = self.wait(lambda: all(self.workers_view().get(w, {}).get("online") for w in self.workers), 60,
                               "registration")
            view = self.workers_view()
            ok = all(view[w]["profile_source"] == "REAL" and view[w]["address"] == f"127.0.0.1:{c['health']}"
                     for w, c in self.workers.items())
            return ok, (f"orchestrator healthy in {t_orch:.2f}s; workers ready (StageExecution listening) in "
                        + ", ".join(f"{w} {s:.1f}s" for w, s in starts.items()) + "; all online+REAL"), {
                "worker_start_seconds": starts, "registration_wait_seconds": waited,
                "registered": {w: {k: view[w].get(k) for k in ("address", "hostname", "profile_source", "online")}
                               for w in self.workers}}

        if not self.step("S02", "start orchestrator + 3 worker.main processes (execution enabled, loopback)", start_all):
            return meta

        def exposure():
            import psutil

            pids = {name: p.proc.pid for name, p in self.procs.items()}
            listeners = []
            for c in psutil.net_connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.pid in pids.values():
                    name = next(n for n, pid in pids.items() if pid == c.pid)
                    listeners.append({"process": name, "pid": c.pid, "address": f"{c.laddr.ip}:{c.laddr.port}"})
            bad = [l for l in listeners if not l["address"].startswith("127.0.0.1:")]
            expected = 2 + 2 * len(self.workers)  # orchestrator REST+gRPC, worker Health+StageExecution
            return (not bad and len(listeners) >= expected,
                    f"{len(listeners)} listening sockets, all on 127.0.0.1" if not bad else f"NON-LOOPBACK: {bad}",
                    {"listeners": sorted(listeners, key=lambda l: l["process"])})

        self.step("S03", "listener exposure audit (psutil, actual bound addresses)", exposure)

        def plan_from_registrations():
            # A worker registers before its backend probe (torch import +
            # compute test, in a child process) finishes; the planner
            # rightly refuses workers whose capabilities are unknown.
            t = time.perf_counter()
            self.wait(lambda: not any(w.get("backend_probe_pending") for w in self.workers_view().values()), 240,
                      "backend probes")
            probe_wait = time.perf_counter() - t
            records = {w.worker_id: w for w in from_orchestrator(self.url)}
            state["live_records"] = records
            plan = self.plan(["val-a", "val-b"], 2, records)
            ev = {"feasible": plan["feasible"], "reasons": (plan.get("infeasibility") or {}).get("reasons"),
                  "usable_ram": {w: records[w].ram_usable_bytes for w in records},
                  "backend_probe_wait_seconds": round(probe_wait, 2)}
            if plan["feasible"]:
                state["records"] = records
                return True, f"feasible from live REAL registrations (waited {probe_wait:.1f}s for backend probes)", ev
            # Honest fallback: the same three REAL processes, but capacity records synthesized for planning only.
            from dataclasses import replace

            state["records"] = {w: replace(r, ram_available_bytes=4 * 2**30, ram_usable_bytes=3 * 2**30,
                                           origin="phase7 validation: synthetic capacity (live data infeasible)")
                                for w, r in records.items()}
            return True, (f"waited {probe_wait:.1f}s for backend probes; INFEASIBLE from live registrations: "
                          f"{'; '.join(ev['reasons'] or [])[:300]} -- pipelines below use these REAL records with a "
                          "SYNTHETIC 3 GiB usable capacity, for planning only"), ev

        self.step("S04", "plan from live orchestrator registrations", plan_from_registrations)
        records = state.get("records")
        plan2 = self.plan(["val-a", "val-b"], 2, records)
        plan3 = self.plan(["val-a", "val-b", "val-c"], 1, records)
        plan3b = self.plan(["val-a", "val-b", "val-c"], 2, records)
        for name, p in (("plan2", plan2), ("plan3", plan3), ("plan3b", plan3b)):
            (self.work / f"{name}.json").write_text(json.dumps(p, indent=1))

        def full_preflight():
            r = preflight.run(orchestrator=self.url, endpoints={w: self.endpoints[w] for w in ("val-a", "val-b")},
                              plan=str(self.work / "plan2.json"), skip_runtime=True)
            fails = [f"{c.section}: {c.name}: {c.detail}" for c in r.checks if c.status == "FAIL"]
            return not fails, f"{r.counts()}" + (f"; FAILS: {fails}" if fails else ""), {"report": r.to_dict()}

        self.step("S05", "preflight: live orchestrator + endpoints + 2-stage plan", full_preflight)

        def pipeline(plan, ids, label):
            def fn():
                r = self.run_remote(plan, ids)
                s = self.summary(r)
                ok = (r["status"] == "succeeded" and r["reference_check"]["passed"]
                      and r["test_classification"] == "REAL_MULTI_PROCESS")
                return ok, (f"{r['status']}, output {s['output_shape']}, next tokens {s['next_token_ids']}, reference diff "
                            f"{(r['reference_check'] or {}).get('max_abs_diff')}, {s['distinct_worker_processes']} processes "
                            f"on {s['distinct_worker_hostnames']}, total {s['elapsed_seconds']:.3f}s"), s
            return fn

        self.step("S06", "2-worker remote pipeline", pipeline(plan2, IDS2, "2"))
        self.step("S07", "3-worker remote pipeline", pipeline(plan3, IDS1, "3"))

        def limit_failure():
            r = self.run_remote(plan3b, IDS2)
            e = r["error"] or {}
            ok = r["status"] == "failed" and e.get("code") == "LIMIT_EXCEEDED" and e.get("executed") is False
            return ok, f"{r['status']} at stage {e.get('stage_index')}: {e.get('code')}: {e.get('problems')}", self.summary(r)

        self.step("S08", "resource-limit failure (val-c max_batch 1, request batch 2)", limit_failure)

        def malformed():
            from partitioner.execution import wire

            pb = wire.pb2()
            # The planner chooses the worker order: read who owns stage 0.
            w0 = plan2["stages"][0]["worker_id"]
            unassigned = f"unassigned stage (stage 1 sent to {w0})"
            cases = {
                "protocol_version 0": self.raw_request(w0, plan2, 0, protocol_version=0),
                "missing plan_json": self.raw_request(w0, plan2, 0, plan_json=b""),
                "corrupt tensor bytes": self.raw_request(w0, plan2, 0,
                                                         _mutate=lambda q: setattr(q.input, "data", q.input.data[:-3])),
                unassigned: self.raw_request(w0, plan2, 1),
                "no deadline": self.raw_request(w0, plan2, 0, _timeout=None),
            }
            want = {"protocol_version 0": "PROTOCOL_VERSION", "missing plan_json": "MALFORMED_REQUEST",
                    "corrupt tensor bytes": "INVALID_TENSOR", unassigned: "UNASSIGNED_STAGE", "no deadline": "DEADLINE"}
            still_ok = self.run_remote(plan2, IDS2)["status"] == "succeeded"
            ok = all(c["status"] == "STAGE_STATUS_REJECTED" and c["code"] == want[k] and not c["executed"]
                     for k, c in cases.items()) and still_ok
            return ok, (f"sent to {w0}: " + ", ".join(f"{k} -> {c['code']}" for k, c in cases.items())
                        + f"; worker still serves afterwards: {still_ok}"), {"cases": cases, "target": w0, "pb_ok": bool(pb)}

        self.step("S09", "malformed requests against a live worker process", malformed)

        def timeout():
            r = self.run_remote(plan2, IDS2, stage_timeout_seconds=0.02, max_attempts=3)
            e = r["error"] or {}
            ok = r["status"] == "failed" and len(e.get("attempts", [])) == 1 and e.get("code") in ("DEADLINE", "AMBIGUOUS_FAILURE")
            return ok, (f"{r['status']}: {e.get('code')} ({(e.get('message') or '')[:120]}); attempts "
                        f"{[a['outcome'] for a in e.get('attempts', [])]} -- not retried"), self.summary(r)

        self.step("S10", "request deadline (20 ms stage timeout)", timeout)

        def graceful_restart():
            p = self.procs["val-b"]
            old_pid = p.proc.pid
            rc = p.graceful_stop()
            clean = any("worker shutdown complete" in l for l in p.text)
            dereg = self.wait(lambda: "val-b" not in self.workers_view(), 10, "deregistration")
            down = self.run_remote(plan2, IDS2, max_attempts=2, connect_timeout_seconds=1, retry_backoff_seconds=0.1)
            t = time.perf_counter()
            self.start_worker("val-b")
            back = self.wait(lambda: self.workers_view().get("val-b", {}).get("online"), 30, "re-registration")
            restart_s = time.perf_counter() - t
            up = self.run_remote(plan2, IDS2)
            new_pid = up["stages"][1]["worker_reported"]["pid"] if up["status"] == "succeeded" else None
            ok = (rc == 0 and clean and down["status"] == "failed" and down["error"]["code"] == "RETRIES_EXHAUSTED"
                  and up["status"] == "succeeded" and new_pid not in (None, old_pid))
            return ok, (f"exit code {rc}, 'worker shutdown complete' logged: {clean}, deregistered in {dereg:.1f}s; while down: "
                        f"{down['status']} {down['error']['code']} at stage {down['error']['stage_index']} (executed "
                        f"{down['error']['executed']}); restarted+online in {restart_s:.1f}s; after: {up['status']}, pid "
                        f"{old_pid} -> {new_pid}"), {"down": self.summary(down), "up": self.summary(up)}

        self.step("S11", "graceful worker stop (CTRL_BREAK) and restart", graceful_restart)

        def hard_kill():
            p = self.procs["val-b"]
            p.kill()
            offline = self.wait(lambda: not self.workers_view()["val-b"]["online"], 20, "offline detection")
            self.start_worker("val-b")
            online = self.wait(lambda: self.workers_view()["val-b"]["online"], 30, "back online")
            up = self.run_remote(plan2, IDS2)
            return (up["status"] == "succeeded",
                    f"killed; orchestrator marked it offline after {offline:.1f}s (heartbeat timeout 3s); restarted, online "
                    f"after {online:.1f}s; pipeline {up['status']}", self.summary(up))

        self.step("S12", "hard kill of a worker, offline detection, restart", hard_kill)

        def orch_restart():
            self.procs["orchestrator"].kill()
            during = self.run_remote(plan2, IDS2)
            t = time.perf_counter()
            self.start_orchestrator("orchestrator-restarted")
            back = self.wait(lambda: all(self.workers_view().get(w, {}).get("online") for w in self.workers), 60,
                             "re-registration after orchestrator restart")
            took = time.perf_counter() - t
            after = self.run_remote(plan2, IDS2)
            relog = {w: any("orchestrator lost registration" in l or "re-registered" in l for l in self.procs[w].text)
                     for w in self.workers}
            ok = during["status"] == "succeeded" and after["status"] == "succeeded" and all(relog.values())
            return ok, (f"pipeline while the orchestrator was DOWN: {during['status']} (execution does not depend on it); "
                        f"after restart all 3 workers re-registered within {took:.1f}s (logged: {relog}); pipeline "
                        f"{after['status']}"), {"during": self.summary(during), "after": self.summary(after)}

        self.step("S13", "orchestrator restart: execution while down, worker re-registration", orch_restart)

        if self.lan_ip:
            self.step("S14", f"LAN-IP bind on this machine ({self.lan_ip})", self.lan_bind)

        def shutdown():
            results = {}
            for wid in self.workers:
                p = self.procs[wid]
                rc = p.graceful_stop()
                results[wid] = {"exit_code": rc, "clean": any("worker shutdown complete" in l for l in p.text)}
            gone = self.wait(lambda: not self.workers_view(), 10, "all deregistered")
            self.procs["orchestrator"].kill()
            ok = all(r["exit_code"] == 0 and r["clean"] for r in results.values())
            return ok, f"{results}; orchestrator lists no workers after {gone:.1f}s", results

        self.step("S15", "clean shutdown of all workers (deregistration)", shutdown)
        meta["finished_at"] = dt.datetime.now().isoformat()
        return meta

    def _timed_start(self, wid):
        t = time.perf_counter()
        self.start_worker(wid)
        return round(time.perf_counter() - t, 2)

    def lan_bind(self):
        """Not run by default (see the module docstring)."""
        port = free_port()
        cfg = self.work / "val-lan.yaml"
        cfg.write_text(f'worker_id: "val-lan"\nexecution_listen_host: "{self.lan_ip}"\n')
        p = Proc("val-lan", [sys.executable, "-m", "worker.execution", "--config", str(cfg), "--port", str(port)],
                 REPO / "worker", self.logs)
        self.procs["val-lan"] = p
        try:
            p.wait_for("STAGE_EXECUTION_LISTENING", 180)
            from partitioner.preflight import tcp_probe

            via_lan = tcp_probe(self.lan_ip, port, 3)
            via_loopback = tcp_probe("127.0.0.1", port, 3)
            ok = via_lan.ok and not via_loopback.ok
            return ok, (f"listening on {self.lan_ip}:{port}; reachable via LAN IP: {via_lan.ok}; via 127.0.0.1: "
                        f"{via_loopback.ok} ({via_loopback.kind}). Traffic to one's own IP stays inside this machine."), {}
        finally:
            p.close()

    def close(self):
        for p in self.procs.values():
            p.close()


def write_report(v: Validation, meta: dict) -> None:
    passed = sum(s["status"] == "PASS" for s in v.steps)
    report = {"phase7_local_validation_schema_version": 1, "meta": meta, "steps": v.steps,
              "summary": {"passed": passed, "failed": len(v.steps) - passed,
                          "classifications": sorted({s["classification"] for s in v.steps})},
              "notes": ["All steps ran on ONE computer; no step is REAL_MULTI_MACHINE.",
                        "Timings are single-run wall clock on a loaded laptop, not benchmarks."]}
    (v.out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [f"# Phase 7 local validation -- {meta.get('started_at', '')[:19]}", "",
             f"Host `{meta.get('hostname')}`, {meta.get('platform')}, Python {meta.get('python')}. "
             "All steps ran on this ONE computer over 127.0.0.1. **No step is REAL_MULTI_MACHINE.** Timings are "
             "single-run wall clock, not benchmarks.", "",
             f"**{passed}/{len(v.steps)} steps passed.**", "", "| Step | Title | Class | Result | Time | Detail |",
             "|---|---|---|---|---|---|"]
    for s in v.steps:
        detail = str(s["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {s['id']} | {s['title']} | {s['classification']} | **{s['status']}** | {s['seconds']:.2f}s | {detail} |")
    lines += ["", "Logs: `logs/` (one file per process). Evidence for every step: `report.json`."]
    (v.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="directory for report.json, report.md and logs/")
    ap.add_argument("--lan-ip", default="", help="also test binding one worker to this private LAN IP (see docstring)")
    args = ap.parse_args()
    if not ORCH_EXE.exists():
        print(f"{ORCH_EXE} not found: cd orchestrator; go build -o orchestrator.exe ./cmd/orchestrator", file=sys.stderr)
        return 2
    v = Validation(Path(args.out).resolve(), args.lan_ip)
    meta = {}
    try:
        meta = v.run()
    finally:
        v.close()
        write_report(v, meta)
    failed = [s["id"] for s in v.steps if s["status"] != "PASS"]
    print(f"\n{len(v.steps) - len(failed)}/{len(v.steps)} steps passed" + (f"; FAILED: {failed}" if failed else ""))
    print(f"report: {v.out / 'report.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
