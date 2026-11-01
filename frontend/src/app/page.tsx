"use client";

/**
 * Phase 1 placeholder Overview page.
 *
 * This deliberately does NOT build out the full 7-page dashboard
 * (docs/ARCHITECTURE.md / master prompt section 7) yet -- that is
 * Phase 9 work. What this page does prove, for real, right now:
 *
 *   - The frontend can reach the Go orchestrator's REST API
 *     (GET /health, GET /api/workers) over a real HTTP call.
 *   - Nothing here is hardcoded or fabricated: with no orchestrator
 *     running, this page honestly shows "unreachable", not a fake
 *     healthy cluster.
 */

import { useEffect, useState } from "react";

const ORCHESTRATOR_REST_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_REST_URL ?? "http://localhost:8080";

type HealthResponse = {
  status: string;
  uptime_seconds: number;
  registered_workers: number;
};

type WorkerView = {
  id: string;
  address: string;
  hostname: string;
  cpu_logical_cores: number;
  total_ram_bytes: number;
  has_cuda_gpu: boolean;
  gpu_name: string;
  os: string;
  status: string;
};

type FetchState =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ok"; health: HealthResponse; workers: WorkerView[] };

function formatBytes(bytes: number): string {
  if (!bytes) return "0 GB";
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export default function OverviewPage() {
  const [state, setState] = useState<FetchState>({ phase: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const [healthRes, workersRes] = await Promise.all([
          fetch(`${ORCHESTRATOR_REST_URL}/health`, { cache: "no-store" }),
          fetch(`${ORCHESTRATOR_REST_URL}/api/workers`, { cache: "no-store" }),
        ]);
        if (!healthRes.ok || !workersRes.ok) {
          throw new Error(`orchestrator returned ${healthRes.status}/${workersRes.status}`);
        }
        const health = (await healthRes.json()) as HealthResponse;
        const workers = (await workersRes.json()) as WorkerView[];
        if (!cancelled) setState({ phase: "ok", health, workers });
      } catch (err) {
        if (!cancelled) {
          setState({
            phase: "error",
            message: err instanceof Error ? err.message : "unknown error",
          });
        }
      }
    }

    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <header>
          <h1 className="text-2xl font-semibold">Decentralized Edge-Compute Pipeline</h1>
          <p className="text-neutral-400 text-sm mt-1">
            Phase 1 connectivity check -- live data from the orchestrator at{" "}
            <code className="text-neutral-300">{ORCHESTRATOR_REST_URL}</code>. The full
            dashboard (worker nodes, model partitioning, pipeline visualization, network
            monitoring, inference playground, logs) is built in Phase 9.
          </p>
        </header>

        {state.phase === "loading" && (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-6 text-neutral-400">
            Connecting to orchestrator...
          </div>
        )}

        {state.phase === "error" && (
          <div className="rounded-lg border border-red-900 bg-red-950/40 p-6">
            <p className="text-red-300 font-medium">Orchestrator unreachable</p>
            <p className="text-red-400/80 text-sm mt-1">{state.message}</p>
            <p className="text-neutral-500 text-sm mt-3">
              Start it with: <code>orchestrator.exe --config orchestrator.yaml</code> (see
              docs/RUNNING_PHASE1.md).
            </p>
          </div>
        )}

        {state.phase === "ok" && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatTile label="Cluster status" value={state.health.status === "ok" ? "Healthy" : state.health.status} />
              <StatTile label="Registered workers" value={String(state.health.registered_workers)} />
              <StatTile label="Orchestrator uptime" value={`${state.health.uptime_seconds.toFixed(0)}s`} />
            </div>

            <section className="rounded-lg border border-neutral-800 bg-neutral-900 overflow-hidden">
              <table className="w-full text-sm text-left">
                <thead className="bg-neutral-950 text-neutral-400 border-b border-neutral-800">
                  <tr>
                    <th className="px-4 py-3">Worker ID</th>
                    <th className="px-4 py-3">Hostname</th>
                    <th className="px-4 py-3">CPU cores</th>
                    <th className="px-4 py-3">RAM</th>
                    <th className="px-4 py-3">GPU</th>
                    <th className="px-4 py-3">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {state.workers.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-6 text-center text-neutral-500">
                        No workers registered yet. Start one with: python -m worker.main --config worker.yaml
                      </td>
                    </tr>
                  )}
                  {state.workers.map((w) => (
                    <tr key={w.id} className="border-b border-neutral-800 last:border-0">
                      <td className="px-4 py-3 font-mono text-neutral-200">{w.id}</td>
                      <td className="px-4 py-3">{w.hostname}</td>
                      <td className="px-4 py-3">{w.cpu_logical_cores}</td>
                      <td className="px-4 py-3">{formatBytes(w.total_ram_bytes)}</td>
                      <td className="px-4 py-3">{w.has_cuda_gpu ? w.gpu_name : "None (CPU-only)"}</td>
                      <td className="px-4 py-3">{w.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </>
        )}
      </div>
    </main>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <p className="text-neutral-500 text-xs uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value}</p>
    </div>
  );
}
