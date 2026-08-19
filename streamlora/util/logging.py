"""Structured, rate-limited logging.

Requirement from the spec: "provide useful diagnostics without flooding logs".
So this module gives two things beyond stdlib logging:

* JSON-lines output, so operational events are queryable after the fact.
* A ``dedupe`` helper that collapses repeated identical events (e.g. a sensor
  that is unavailable on every single tick) into periodic summaries.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(record.created))
        base = f"{t} {record.levelname[0]} {record.name:24s} {record.getMessage()}"
        extra = getattr(record, "fields", None)
        if extra:
            base += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_configured = False
_lock = threading.Lock()


def configure(level: str = "info", fmt: str = "human", path: str | None = None) -> None:
    global _configured
    with _lock:
        root = logging.getLogger("streamlora")
        root.handlers.clear()
        root.setLevel(_LEVELS.get(level, logging.INFO))
        root.propagate = False
        formatter = JsonLineFormatter() if fmt == "json" else HumanFormatter()
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)
        if path:
            fh = logging.FileHandler(path)
            fh.setFormatter(JsonLineFormatter())
            root.addHandler(fh)
        _configured = True


def get_logger(name: str) -> "EventLogger":
    if not _configured:
        configure()
    return EventLogger(logging.getLogger(f"streamlora.{name}"))


class EventLogger:
    """Thin wrapper giving ``log.info("msg", key=value)`` plus dedupe."""

    def __init__(self, inner: logging.Logger) -> None:
        self._log = inner
        self._seen: dict[str, tuple[float, int]] = {}

    def _emit(self, level: int, msg: str, fields: dict[str, Any]) -> None:
        self._log.log(level, msg, extra={"fields": fields} if fields else None)

    def debug(self, msg: str, **f: Any) -> None:
        self._emit(logging.DEBUG, msg, f)

    def info(self, msg: str, **f: Any) -> None:
        self._emit(logging.INFO, msg, f)

    def warning(self, msg: str, **f: Any) -> None:
        self._emit(logging.WARNING, msg, f)

    def error(self, msg: str, **f: Any) -> None:
        self._emit(logging.ERROR, msg, f)

    def exception(self, msg: str, **f: Any) -> None:
        self._log.log(logging.ERROR, msg, exc_info=True, extra={"fields": f} if f else None)

    def dedupe(
        self,
        key: str,
        msg: str,
        level: str = "warning",
        every: float = 300.0,
        now: float | None = None,
        **f: Any,
    ) -> None:
        """Log at most once per ``every`` seconds per ``key``.

        Suppressed occurrences are counted and reported on the next emission,
        so a permanently missing sensor produces a handful of lines per hour
        instead of one per tick, without losing the count.
        """
        t = time.time() if now is None else now
        entry = self._seen.get(key)
        if entry is None:
            # Always emit the first occurrence. Using 0.0 as a sentinel would
            # silently suppress it whenever the caller passes simulated
            # timestamps that start near zero (replays, tests).
            self._emit(_LEVELS.get(level, logging.WARNING), msg, f)
            self._seen[key] = (t, 0)
            return
        last, count = entry
        if t - last >= every:
            if count:
                f["suppressed"] = count
            self._emit(_LEVELS.get(level, logging.WARNING), msg, f)
            self._seen[key] = (t, 0)
        else:
            self._seen[key] = (last, count + 1)
