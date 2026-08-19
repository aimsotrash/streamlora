"""Deterministic-friendly identifier helpers."""

from __future__ import annotations

import hashlib
import itertools
import os

_counter = itertools.count(1)


def short_uid(prefix: str = "") -> str:
    raw = f"{os.getpid()}-{next(_counter)}".encode()
    h = hashlib.blake2s(raw, digest_size=5).hexdigest()
    return f"{prefix}{h}" if prefix else h


def version_label(kind: str, n: int) -> str:
    """``version_label('forecast-model', 2) -> 'forecast-model-v002'``."""
    return f"{kind}-v{n:03d}"


def config_hash(obj: object) -> str:
    """Stable hash of a JSON-able config, used to tie results to inputs."""
    import json

    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2s(blob.encode(), digest_size=8).hexdigest()
