"""Run ONLY the StageExecution service (no registration, no heartbeats):

    python -m worker.execution --config worker.yaml [--port 0]

Used by the two-process integration tests and handy for operators.
`python -m worker.main` starts the same service alongside registration
when execution_enabled is true. Prints one machine-readable line once
listening:

    STAGE_EXECUTION_LISTENING <host>:<port> worker_id=<id>
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from worker.config.loader import load as load_config
from worker.execution.server import start_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="StageExecution service only (Phase 6)")
    parser.add_argument("--config", default="worker.yaml")
    parser.add_argument("--port", type=int, default=None, help="override execution_listen_port (0 = any free port)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, lambda signum, frame: stop.set())
            except (ValueError, OSError):
                pass
    server, address = start_from_config(cfg, port=args.port)
    print(f"STAGE_EXECUTION_LISTENING {address} worker_id={cfg.worker_id}", flush=True)
    try:
        while not stop.wait(0.5):
            pass
    finally:
        server.stop(grace=2).wait(3)
        logging.getLogger("worker.execution").info("StageExecution stopped")


if __name__ == "__main__":
    main()
