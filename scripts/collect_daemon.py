"""Standalone collector process.

Used to accumulate real telemetry in the background during development. The
shipped entry point is ``streamlora collect``; this script exists so collection
can be started before the CLI's other subcommands are wired up, and so it can
be supervised independently of any interactive session.
"""
from __future__ import annotations

import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlora.config import Config
from streamlora.pipeline import build_telemetry_pipeline
from streamlora.util.ids import config_hash
from streamlora.util.logging import configure, get_logger


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    cfg = Config()
    cfg.general.data_dir = data_dir
    cfg.general.log_format = "json"
    cfg.general.log_file = os.path.join(data_dir, "collect.log.jsonl")
    cfg.collect.interval_s = interval
    cfg.ensure_dirs()
    configure(cfg.general.log_level, cfg.general.log_format, cfg.general.log_file)
    log = get_logger("collect_daemon")

    run_id = f"live-{time.strftime('%Y%m%d-%H%M%S')}"
    pipe = build_telemetry_pipeline(cfg, run_id=run_id, origin="live")
    pipe.repos.runs.start(
        run_id, "collect", time.time(), cfg.to_dict(), config_hash(cfg.to_dict()),
        notes=f"background collection, {hours}h at {interval}s",
    )
    with open(os.path.join(data_dir, "collector.pid"), "w") as fh:
        fh.write(str(os.getpid()))

    stopping = {"flag": False}

    def handle(signum, _frame):
        stopping["flag"] = True
        pipe.collector.stop()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    log.info("collection started", run_id=run_id, hours=hours, interval_s=interval)
    try:
        pipe.collector.run(duration_s=hours * 3600.0)
    finally:
        n = pipe.repos.telemetry.count(run_id=run_id)
        pipe.repos.runs.finish(
            run_id, time.time(), "stopped" if stopping["flag"] else "complete",
            summary={"samples": n, **pipe.collector.stats.as_dict()},
        )
        log.info("collection finished", run_id=run_id, samples=n)
        pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
