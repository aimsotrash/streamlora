"""Temperature sensors, mapped from platform-specific chips to portable roles.

``psutil.sensors_temperatures()`` returns whatever the kernel exposes, keyed by
driver name: ``k10temp`` on AMD, ``coretemp`` on Intel, ``amdgpu``/``nvidia`` for
graphics, ``nvme`` for storage. Emitting those raw names as signals would make
every stored dataset, config and trained model machine-specific.

So this source maps drivers onto a small set of roles (cpu / gpu / disk /
wireless) plus ``thermal.max_c``. A model trained on one laptop still has
meaningful feature names on another, and an unmapped driver degrades to
contributing only to ``thermal.max_c`` rather than being silently dropped.
"""

from __future__ import annotations

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource

#: driver-name prefix -> role. Checked longest-prefix-first.
_ROLE_MAP: dict[str, str] = {
    "k10temp": "cpu",
    "coretemp": "cpu",
    "zenpower": "cpu",
    "cpu_thermal": "cpu",
    "acpitz": "cpu",
    "soc_thermal": "cpu",
    "amdgpu": "gpu",
    "nvidia": "gpu",
    "radeon": "gpu",
    "i915": "gpu",
    "nvme": "disk",
    "drivetemp": "disk",
    "iwlwifi": "wireless",
    "mt76": "wireless",
    "ath": "wireless",
}

_ROLES = ("cpu", "gpu", "disk", "wireless")

_SPECS = [
    SignalSpec(f"thermal.{role}_c", "celsius", SignalKind.GAUGE, -20.0, 130.0,
               max_rate_per_s=15.0,
               description=f"{role} temperature (highest sensor mapped to this role)")
    for role in _ROLES
] + [
    SignalSpec("thermal.max_c", "celsius", SignalKind.GAUGE, -20.0, 130.0,
               max_rate_per_s=15.0,
               description="Hottest sensor anywhere on the machine, including unmapped drivers"),
]


def _role_for(driver: str) -> str | None:
    d = driver.lower()
    for prefix in sorted(_ROLE_MAP, key=len, reverse=True):
        if d.startswith(prefix):
            return _ROLE_MAP[prefix]
    return None


class ThermalSource(TelemetrySource):
    name = "thermal"
    description = "CPU / GPU / disk / wireless temperatures"

    def __init__(self) -> None:
        self._roles_found: tuple[str, ...] = ()
        self._drivers: dict[str, str | None] = {}

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        try:
            temps = psutil.sensors_temperatures()
        except Exception as exc:
            return ProbeResult(False, f"sensors_temperatures failed: {exc}")
        if not temps:
            return ProbeResult(False, "no temperature sensors exposed")
        roles: set[str] = set()
        self._drivers = {}
        for driver in temps:
            role = _role_for(driver)
            self._drivers[driver] = role
            if role:
                roles.add(role)
        self._roles_found = tuple(r for r in _ROLES if r in roles)
        provides = tuple(f"thermal.{r}_c" for r in self._roles_found) + ("thermal.max_c",)
        unmapped = [d for d, r in self._drivers.items() if r is None]
        detail = f"drivers={sorted(temps)}"
        if unmapped:
            detail += f" unmapped={unmapped}"
        return ProbeResult(True, detail, provides)

    def read(self, now: float) -> dict[str, float | None]:
        out: dict[str, float | None] = {f"thermal.{r}_c": None for r in self._roles_found}
        out["thermal.max_c"] = None
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            return out
        best: dict[str, float] = {}
        overall: float | None = None
        for driver, entries in temps.items():
            role = self._drivers.get(driver, _role_for(driver))
            for e in entries:
                cur = getattr(e, "current", None)
                if cur is None:
                    continue
                cur = float(cur)
                # Drivers occasionally report 0.0 for an unpopulated channel.
                if cur <= 0.0:
                    continue
                overall = cur if overall is None else max(overall, cur)
                if role:
                    best[role] = max(best.get(role, cur), cur)
        for role, v in best.items():
            out[f"thermal.{role}_c"] = v
        out["thermal.max_c"] = overall
        return out
