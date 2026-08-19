"""Battery state of charge, charge direction and instantaneous power draw.

Two mechanisms, in preference order:

1. Linux ``/sys/class/power_supply``. Preferred because ``charge_now /
   charge_full`` (or the energy_* equivalents) gives *fractional* state of
   charge. The integer ``capacity`` file -- and psutil, which reports the same
   number -- quantises to 1%, which is fatal for short-horizon discharge-rate
   estimation: at a realistic 0.15 %/min drain, a 1% quantum means the signal
   is a staircase with ~7 minute treads and the instantaneous slope is either
   zero or a spike. Fractional charge turns that into a usable gradient.
2. ``psutil.sensors_battery()``. Cross-platform fallback (macOS, Windows,
   BSD), coarser but always structurally the same signals.

A machine with no battery reports the source unavailable and everything else
keeps working -- that is an explicit requirement, not an accident.
"""

from __future__ import annotations

import os

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource, read_sysfs_float

_PS_ROOT = "/sys/class/power_supply"

_SPECS = [
    SignalSpec(
        "battery.percent", "percent", SignalKind.GAUGE, 0.0, 100.0,
        # A battery cannot plausibly move more than ~0.5 %/s (that would be a
        # 200 s full discharge). Anything faster is a sensor glitch.
        max_rate_per_s=0.5, max_hold_s=120.0,
        description="State of charge, fractional where the platform exposes it",
    ),
    SignalSpec(
        "battery.plugged", "bool", SignalKind.FLAG, 0.0, 1.0, max_hold_s=120.0,
        description="1 when external power is connected",
    ),
    SignalSpec(
        "battery.power_w", "watt", SignalKind.GAUGE, 0.0, 400.0, max_hold_s=120.0,
        description="Magnitude of battery charge/discharge power",
    ),
    SignalSpec(
        "battery.charging", "bool", SignalKind.FLAG, 0.0, 1.0, max_hold_s=120.0,
        description="1 while the pack is actually taking charge (plugged but full reads 0)",
    ),
]


def _find_battery_dir() -> str | None:
    try:
        names = sorted(os.listdir(_PS_ROOT))
    except OSError:
        return None
    for n in names:
        d = os.path.join(_PS_ROOT, n)
        if read_sysfs_float(os.path.join(d, "capacity")) is not None:
            return d
        # Some packs expose energy/charge but not capacity.
        if os.path.exists(os.path.join(d, "charge_now")) or os.path.exists(
            os.path.join(d, "energy_now")
        ):
            return d
    return None


def _find_ac_dirs() -> list[str]:
    try:
        names = sorted(os.listdir(_PS_ROOT))
    except OSError:
        return []
    out = []
    for n in names:
        d = os.path.join(_PS_ROOT, n)
        try:
            with open(os.path.join(d, "type")) as fh:
                if fh.read().strip() == "Mains":
                    out.append(d)
        except OSError:
            continue
    return out


class BatterySource(TelemetrySource):
    name = "battery"
    description = "State of charge, AC presence and charge/discharge power"

    def __init__(self) -> None:
        self._bat: str | None = None
        self._ac: list[str] = []
        self._mode = "none"

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        self._bat = _find_battery_dir()
        self._ac = _find_ac_dirs()
        if self._bat is not None:
            self._mode = "sysfs"
            frac = self._sysfs_fraction() is not None
            provides = ["battery.percent", "battery.plugged", "battery.charging"]
            if self._sysfs_power() is not None:
                provides.append("battery.power_w")
            detail = f"sysfs {os.path.basename(self._bat)}"
            detail += ", fractional charge" if frac else ", integer capacity only"
            return ProbeResult(True, detail, tuple(provides))
        try:
            if psutil.sensors_battery() is not None:
                self._mode = "psutil"
                return ProbeResult(
                    True, "psutil (integer percent)",
                    ("battery.percent", "battery.plugged", "battery.charging"),
                )
        except Exception as exc:
            return ProbeResult(False, f"psutil battery probe failed: {exc}")
        self._mode = "none"
        return ProbeResult(False, "no battery detected")

    # -- sysfs helpers -----------------------------------------------------
    def _sysfs_fraction(self) -> float | None:
        """Fractional state of charge in percent, or None."""
        d = self._bat
        if d is None:
            return None
        for now_f, full_f in (("charge_now", "charge_full"), ("energy_now", "energy_full")):
            now = read_sysfs_float(os.path.join(d, now_f))
            full = read_sysfs_float(os.path.join(d, full_f))
            if now is not None and full is not None and full > 0:
                return 100.0 * now / full
        return None

    def _sysfs_power(self) -> float | None:
        """Instantaneous pack power in watts, or None."""
        d = self._bat
        if d is None:
            return None
        p = read_sysfs_float(os.path.join(d, "power_now"), 1e-6)  # microwatt -> W
        if p is not None:
            return abs(p)
        cur = read_sysfs_float(os.path.join(d, "current_now"), 1e-6)  # uA -> A
        volt = read_sysfs_float(os.path.join(d, "voltage_now"), 1e-6)  # uV -> V
        if cur is not None and volt is not None:
            return abs(cur * volt)
        return None

    def _sysfs_status(self) -> str | None:
        d = self._bat
        if d is None:
            return None
        try:
            with open(os.path.join(d, "status")) as fh:
                return fh.read().strip()
        except OSError:
            return None

    def _sysfs_ac_online(self) -> float | None:
        for d in self._ac:
            v = read_sysfs_float(os.path.join(d, "online"))
            if v is not None:
                return 1.0 if v > 0 else 0.0
        return None

    # -- read --------------------------------------------------------------
    def read(self, now: float) -> dict[str, float | None]:
        if self._mode == "sysfs":
            return self._read_sysfs()
        if self._mode == "psutil":
            return self._read_psutil()
        return dict.fromkeys(("battery.percent", "battery.plugged", "battery.power_w", "battery.charging"))

    def _read_sysfs(self) -> dict[str, float | None]:
        d = self._bat
        assert d is not None
        pct = self._sysfs_fraction()
        if pct is None:
            pct = read_sysfs_float(os.path.join(d, "capacity"))
        status = self._sysfs_status()
        ac = self._sysfs_ac_online()
        if ac is None and status is not None:
            # Infer AC from status when no Mains device is exposed.
            ac = 0.0 if status == "Discharging" else 1.0
        charging = None
        if status is not None:
            charging = 1.0 if status == "Charging" else 0.0
        return {
            "battery.percent": pct,
            "battery.plugged": ac,
            "battery.power_w": self._sysfs_power(),
            "battery.charging": charging,
        }

    def _read_psutil(self) -> dict[str, float | None]:
        try:
            b = psutil.sensors_battery()
        except Exception:
            b = None
        if b is None:
            return dict.fromkeys(
                ("battery.percent", "battery.plugged", "battery.power_w", "battery.charging")
            )
        plugged = None if b.power_plugged is None else (1.0 if b.power_plugged else 0.0)
        # psutil cannot distinguish "charging" from "plugged and full"; treat
        # plugged-and-not-full as charging, which is right on every laptop that
        # does not implement a charge threshold.
        charging = None
        if plugged is not None:
            charging = 1.0 if (plugged > 0 and b.percent < 99.5) else 0.0
        return {
            "battery.percent": float(b.percent),
            "battery.plugged": plugged,
            "battery.power_w": None,
            "battery.charging": charging,
        }
