"""Telemetry: collectors, degraded sensors, malformed data, gaps, normalisation."""

from __future__ import annotations

import math

import pytest

from streamlora.telemetry.collector import Collector
from streamlora.telemetry.normalizer import Normalizer
from streamlora.telemetry.registry import SourceRegistry, default_sources
from streamlora.telemetry.schema import (
    Quality,
    Reading,
    SignalKind,
    SignalRegistry,
    SignalSpec,
    TelemetrySample,
)
from streamlora.telemetry.sources.base import ProbeResult, RateTracker, TelemetrySource
from streamlora.util.clock import SimulatedClock

from .conftest import T0


# --------------------------------------------------------------------------
# fake sources
# --------------------------------------------------------------------------

class GoodSource(TelemetrySource):
    name = "good"

    def __init__(self, value: float = 10.0) -> None:
        self.value = value
        self.reads = 0

    def signals(self):
        return [SignalSpec("fake.value", "percent", SignalKind.GAUGE, 0.0, 100.0)]

    def probe(self):
        return ProbeResult(True, "fake", ("fake.value",))

    def read(self, now):
        self.reads += 1
        return {"fake.value": self.value}


class MissingSource(TelemetrySource):
    """A sensor that exists but has nothing to report."""

    name = "missing"

    def signals(self):
        return [SignalSpec("fake.absent", "percent", SignalKind.GAUGE, 0.0, 100.0)]

    def probe(self):
        return ProbeResult(True, "present but empty", ("fake.absent",))

    def read(self, now):
        return {"fake.absent": None}


class UnavailableSource(TelemetrySource):
    name = "unavailable"

    def signals(self):
        return [SignalSpec("fake.never", "percent")]

    def probe(self):
        return ProbeResult(False, "no such device")

    def read(self, now):  # pragma: no cover - never called
        raise AssertionError("unavailable source must not be read")


class ExplodingSource(TelemetrySource):
    name = "exploding"

    def __init__(self, fail_after: int = 0) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def signals(self):
        return [SignalSpec("fake.boom", "percent", SignalKind.GAUGE, 0.0, 100.0)]

    def probe(self):
        return ProbeResult(True, "will fail", ("fake.boom",))

    def read(self, now):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("sensor exploded")
        return {"fake.boom": 1.0}


class ProbeRaises(TelemetrySource):
    name = "probe_raises"

    def signals(self):
        return [SignalSpec("fake.p", "percent")]

    def probe(self):
        raise RuntimeError("probe blew up")

    def read(self, now):  # pragma: no cover
        return {}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_skips_unavailable_sources(clock):
    reg = SourceRegistry([GoodSource(), UnavailableSource()], clock)
    health = reg.probe_all()
    assert health["good"].available
    assert not health["unavailable"].available
    assert "fake.never" not in reg.signals
    values, errors = reg.read_all(T0)
    assert values["fake.value"] == 10.0
    assert not errors


def test_registry_survives_a_probe_that_raises(clock):
    reg = SourceRegistry([ProbeRaises(), GoodSource()], clock)
    health = reg.probe_all()
    assert not health["probe_raises"].available
    assert health["good"].available


def test_a_failing_source_does_not_stop_the_others(clock):
    good = GoodSource()
    bad = ExplodingSource(fail_after=0)
    reg = SourceRegistry([good, bad], clock)
    reg.probe_all()
    reported = 0
    for i in range(5):
        values, errors = reg.read_all(T0 + i * 5)
        # The requirement from the brief: a missing battery must never stop CPU
        # forecasting. The healthy source keeps producing on every tick.
        assert values["fake.value"] == 10.0
        # The broken one still appears in the sample, as an explicit None.
        assert "fake.boom" in values and values["fake.boom"] is None
        reported += "exploding" in errors
    # Failures are reported until the source is quarantined, then it is skipped
    # entirely rather than re-failing every tick.
    assert 3 <= reported <= 5
    assert reg.health["exploding"].failures >= 3
    assert reg.health["exploding"].quarantined_until > 0


def test_quarantined_source_is_not_read_until_backoff_expires(clock):
    bad = ExplodingSource(fail_after=0)
    reg = SourceRegistry([bad], clock)
    reg.probe_all()
    for i in range(3):
        reg.read_all(T0 + i)
    calls_at_quarantine = bad.calls
    assert reg.health["exploding"].quarantined_until > T0 + 2
    reg.read_all(T0 + 3)          # inside the backoff window
    assert bad.calls == calls_at_quarantine
    reg.read_all(T0 + 600)        # well past it
    assert bad.calls == calls_at_quarantine + 1


def test_source_declaring_a_cadence_is_decimated(clock):
    src = GoodSource()
    src.min_interval_s = 30.0
    reg = SourceRegistry([src], clock)
    reg.probe_all()
    reg.read_all(T0)
    assert src.reads == 1
    for i in range(1, 6):
        reg.read_all(T0 + i * 5)   # 5..25 s: all inside the cadence
    assert src.reads == 1
    reg.read_all(T0 + 35)
    assert src.reads == 2


def test_default_sources_construct_without_raising():
    # Construction must never depend on the host having any given device.
    srcs = default_sources(None)
    assert srcs
    assert all(hasattr(s, "probe") for s in srcs)


# --------------------------------------------------------------------------
# normalizer
# --------------------------------------------------------------------------

def test_missing_value_is_held_then_marked_missing(simple_registry):
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    n.normalize(T0, {"cpu.util_pct": 40.0})
    s = n.normalize(T0 + 5, {})
    assert s.readings["cpu.util_pct"].value == 40.0
    assert s.readings["cpu.util_pct"].quality is Quality.STALE
    s = n.normalize(T0 + 100, {})   # past max_hold_s (30 s default)
    assert s.readings["cpu.util_pct"].value is None
    assert s.readings["cpu.util_pct"].quality is Quality.MISSING


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "abc", None])
def test_malformed_values_never_reach_a_model(simple_registry, bad):
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    s = n.normalize(T0, {"cpu.util_pct": bad})
    r = s.readings["cpu.util_pct"]
    assert r.value is None or math.isfinite(r.value)
    assert not (r.value is not None and r.quality is Quality.OK and not math.isfinite(r.value))


def test_out_of_range_is_clamped_and_flagged(simple_registry):
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    s = n.normalize(T0, {"cpu.util_pct": 150.0})
    assert s.readings["cpu.util_pct"].value == 100.0
    assert s.readings["cpu.util_pct"].quality is Quality.CLAMPED
    s = n.normalize(T0 + 5, {"cpu.util_pct": -3.0})
    assert s.readings["cpu.util_pct"].value == 0.0


def test_sub_unit_overshoot_is_clipped_without_flagging(simple_registry):
    """psutil reports 100.4% routinely; that is rounding, not a fault."""
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    s = n.normalize(T0, {"cpu.util_pct": 100.0000001})
    assert s.readings["cpu.util_pct"].quality is Quality.OK


def test_implausible_rate_of_change_is_flagged_suspect(simple_registry):
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    n.normalize(T0, {"battery.percent": 80.0})
    s = n.normalize(T0 + 5, {"battery.percent": 20.0})   # 12 %/s, limit is 0.5
    assert s.readings["battery.percent"].quality is Quality.SUSPECT
    assert s.readings["battery.percent"].value == 20.0   # kept, not dropped


def test_gap_detection_and_hold_state_reset(simple_registry):
    n = Normalizer(simple_registry, expected_interval_s=5.0, gap_factor=3.0)
    n.normalize(T0, {"cpu.util_pct": 40.0})
    s = n.normalize(T0 + 5, {"cpu.util_pct": 41.0})
    assert s.gap_s == pytest.approx(5.0)
    s = n.normalize(T0 + 4000, {})       # suspend/resume
    assert s.gap_s == pytest.approx(3995.0)
    # Holding a value across a four-hour gap would be a lie.
    assert s.readings["cpu.util_pct"].quality is Quality.MISSING
    assert n.stats.gaps == 1


def test_timestamps_are_recorded_and_monotone(simple_registry):
    n = Normalizer(simple_registry, expected_interval_s=5.0)
    out = [n.normalize(T0 + i * 5, {"cpu.util_pct": float(i)}) for i in range(5)]
    assert [s.ts for s in out] == sorted(s.ts for s in out)
    assert out[0].gap_s is None


def test_worst_quality_aggregates():
    s = TelemetrySample(ts=T0, readings={
        "a": Reading(1.0, Quality.OK), "b": Reading(None, Quality.MISSING),
    })
    assert s.worst_quality is Quality.MISSING
    assert s.usable_signals() == ["a"]


# --------------------------------------------------------------------------
# rate tracker
# --------------------------------------------------------------------------

def test_rate_tracker_first_read_and_reset():
    rt = RateTracker()
    assert rt.update(1000, T0) is None            # no rate on first observation
    assert rt.update(2000, T0 + 10) == pytest.approx(100.0)
    assert rt.update(500, T0 + 20) is None        # counter reset -> skip interval
    assert rt.update(1500, T0 + 30) == pytest.approx(100.0)


def test_rate_tracker_rejects_non_positive_dt():
    rt = RateTracker()
    rt.update(0, T0)
    assert rt.update(100, T0) is None


# --------------------------------------------------------------------------
# collector
# --------------------------------------------------------------------------

def test_collector_batches_writes_and_calls_handlers(clock, simple_registry):
    reg = SourceRegistry([GoodSource()], clock)
    reg.probe_all()
    norm = Normalizer(reg.signals, expected_interval_s=5.0)
    written: list[list] = []
    seen: list = []
    c = Collector(reg, norm, clock=clock, interval_s=5.0, sink=written.append, write_batch=3)
    c.subscribe(seen.append)
    c.run_for_ticks(7)
    assert len(seen) == 7
    assert sum(len(b) for b in written) == 7
    assert clock.now() == pytest.approx(T0 + 35)


def test_collector_survives_a_broken_handler(clock):
    reg = SourceRegistry([GoodSource()], clock)
    reg.probe_all()
    norm = Normalizer(reg.signals, expected_interval_s=5.0)
    c = Collector(reg, norm, clock=clock, interval_s=5.0)

    def boom(_s):
        raise ValueError("handler is broken")

    c.subscribe(boom)
    c.run_for_ticks(3)
    # Telemetry is the one thing that cannot be recovered retroactively, so a
    # broken subscriber must never stop collection.
    assert c.stats.ticks == 3
    assert c.stats.handler_errors == 3


def test_collector_realigns_after_an_overrun(clock):
    reg = SourceRegistry([GoodSource()], clock)
    reg.probe_all()
    norm = Normalizer(reg.signals, expected_interval_s=5.0)
    c = Collector(reg, norm, clock=clock, interval_s=5.0)
    c.tick()
    c._sleep_to_next_tick()      # establishes the grid
    c.tick()
    clock.advance(23.0)          # work took far longer than the interval
    c._sleep_to_next_tick()
    assert c.stats.overruns == 1
    assert c.stats.skipped_ticks >= 1
    # The grid must stay aligned rather than drift or burst to catch up.
    assert (clock.now() - T0) % 5.0 == pytest.approx(0.0, abs=1e-9)
