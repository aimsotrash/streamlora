"""Workload composition, expressed as CPU share per activity category.

This is the source that lets the system distinguish "70% CPU because of a
build" from "70% CPU because of a video call" -- the distinction that makes
regime-aware forecasting and personalised explanations possible.

Privacy decision
----------------
Process *names* are markedly more sensitive than aggregate utilisation: they
reveal which applications and projects a person uses. This source therefore
aggregates into coarse categories and, by default, persists **only the
category shares** -- never process names or command lines. ``include_names``
exists for local debugging and is off by default.

The category table is a starting point, not ground truth. It is data, not
control flow, so it can be edited or learned later without touching the
pipeline (see docs/architecture.md on regime evolution).
"""

from __future__ import annotations

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource

#: category -> substrings matched against the lowercased process name.
CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "build": (
        "cc1", "cc1plus", "gcc", "g++", "clang", "clang++", "ld", "lld", "rustc",
        "cargo", "ninja", "make", "cmake", "javac", "kotlinc", "scalac", "tsc",
        "webpack", "esbuild", "rollup", "vite", "gradle", "mvn", "bazel", "swiftc",
        "conda-build", "pip", "setup.py", "meson", "sccache", "ccache",
    ),
    "browser": (
        "firefox", "chrome", "chromium", "brave", "msedge", "safari", "webkit",
        "vivaldi", "opera", "librewolf",
    ),
    "media": (
        "vlc", "mpv", "ffmpeg", "obs", "spotify", "handbrake", "gstreamer",
        "pipewire", "wireplumber", "totem", "celluloid",
    ),
    "ml": (
        "python3", "python", "jupyter", "ollama", "llama", "torchrun", "tensorboard",
        "vllm", "ray",
    ),
    "container": ("docker", "containerd", "podman", "crio", "qemu", "virtqemud", "libvirt"),
    "editor": ("code", "codium", "cursor", "nvim", "vim", "emacs", "idea", "pycharm", "zed", "sublime"),
    "system": (
        "systemd", "kworker", "dbus", "journald", "packagekit", "dnf", "rpm", "apt",
        "gnome-shell", "cosmic", "kwin", "plasmashell", "Xorg", "wayland",
    ),
}

_CATEGORIES = tuple(CATEGORY_PATTERNS)

_SPECS = [
    SignalSpec("proc.count", "count", SignalKind.GAUGE, 0.0, 100000.0,
               description="Number of processes visible to this user"),
    SignalSpec("proc.top1_cpu_pct", "percent", SignalKind.GAUGE, 0.0, 6400.0,
               description="CPU percent of the single busiest process (can exceed 100 across cores)"),
    SignalSpec("proc.concentration", "ratio", SignalKind.GAUGE, 0.0, 1.0,
               description="Share of total process CPU held by the top 3 processes; 1.0 means one dominant task"),
] + [
    SignalSpec(f"proc.cpu_{cat}_pct", "percent", SignalKind.GAUGE, 0.0, 6400.0,
               description=f"Summed CPU percent of processes categorised as {cat}")
    for cat in _CATEGORIES
]


def categorise(name: str) -> str | None:
    n = name.lower()
    for cat, pats in CATEGORY_PATTERNS.items():
        for p in pats:
            if p in n:
                return cat
    return None


class ProcessSource(TelemetrySource):
    name = "process"
    description = "Per-category CPU attribution (aggregate only; no process names stored)"

    #: A full pass over ~500 processes costs ~40 ms. At the default 5 s cadence
    #: that is under 1% duty cycle, so we sample every tick -- and psutil's
    #: per-process cpu_percent is only meaningful with a consistent interval.
    min_interval_s: float = 0.0

    def __init__(self, include_names: bool = False, top_n: int = 3) -> None:
        self.include_names = include_names
        self.top_n = top_n
        self._cache: dict[int, psutil.Process] = {}
        #: Populated each read when ``include_names`` is set. Never persisted by
        #: the collector; exposed only for interactive debugging.
        self.last_top: list[tuple[str, float]] = []

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        try:
            pids = psutil.pids()
        except Exception as exc:
            return ProbeResult(False, f"pids() failed: {exc}")
        self._refresh_cache(pids)
        # Prime the per-process CPU deltas so the first real read is valid.
        for p in list(self._cache.values()):
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        return ProbeResult(True, f"{len(pids)} processes visible", tuple(s.name for s in _SPECS))

    def _refresh_cache(self, pids: list[int]) -> None:
        live = set(pids)
        for pid in live - self._cache.keys():
            try:
                self._cache[pid] = psutil.Process(pid)
            except psutil.Error:
                continue
        for pid in self._cache.keys() - live:
            self._cache.pop(pid, None)

    def read(self, now: float) -> dict[str, float | None]:
        out: dict[str, float | None] = dict.fromkeys((s.name for s in _SPECS), None)
        try:
            pids = psutil.pids()
        except Exception:
            return out
        self._refresh_cache(pids)
        by_cat: dict[str, float] = dict.fromkeys(_CATEGORIES, 0.0)
        cpus: list[tuple[float, str]] = []
        total = 0.0
        for pid, proc in list(self._cache.items()):
            try:
                cpu = proc.cpu_percent(None)
                name = proc.name()
            except psutil.Error:
                # Process exited between pid listing and read: expected churn.
                self._cache.pop(pid, None)
                continue
            if cpu <= 0.0:
                continue
            total += cpu
            cpus.append((cpu, name))
            cat = categorise(name)
            if cat:
                by_cat[cat] += cpu
        cpus.sort(reverse=True)
        out["proc.count"] = float(len(self._cache))
        out["proc.top1_cpu_pct"] = cpus[0][0] if cpus else 0.0
        top_sum = sum(c for c, _ in cpus[: self.top_n])
        out["proc.concentration"] = (top_sum / total) if total > 0 else 0.0
        for cat, v in by_cat.items():
            out[f"proc.cpu_{cat}_pct"] = v
        self.last_top = [(n, c) for c, n in cpus[: self.top_n]] if self.include_names else []
        return out
