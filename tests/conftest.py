"""Shared fixtures.

Every test that needs storage gets an in-memory database and a simulated clock,
so the suite never touches the developer's real telemetry and never depends on
wall-clock timing.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlora.config import Config
from streamlora.replay.synthetic import build_scenario, generate, synthetic_registry
from streamlora.store.repo import Repos
from streamlora.telemetry.schema import Reading, SignalKind, SignalRegistry, SignalSpec, TelemetrySample
from streamlora.util.clock import SimulatedClock
from streamlora.util.logging import configure

configure("error", "human")

T0 = 1_760_000_000.0


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(T0)


@pytest.fixture
def repos(tmp_path):
    r = Repos.open(str(tmp_path / "test.sqlite"))
    yield r
    r.close()


@pytest.fixture
def config(tmp_path) -> Config:
    c = Config()
    c.general.data_dir = str(tmp_path)
    c.collect.interval_s = 5.0
    c.forecast.horizons_s = [300.0, 900.0]
    c.forecast.standardize_warmup = 40
    c.forecast.min_train_before_predict = 10
    c.adapt.every_n_samples = 60
    c.adapt.gate_window = 40
    c.adapt.gate_min_samples = 20
    c.uncertainty.min_residuals = 20
    c.uncertainty.window = 200
    c.ensure_dirs()
    return c


@pytest.fixture
def simple_registry() -> SignalRegistry:
    return SignalRegistry([
        SignalSpec("cpu.util_pct", "percent", SignalKind.GAUGE, 0.0, 100.0),
        SignalSpec("mem.used_pct", "percent", SignalKind.GAUGE, 0.0, 100.0),
        SignalSpec("battery.percent", "percent", SignalKind.GAUGE, 0.0, 100.0,
                   max_rate_per_s=0.5, max_hold_s=120.0),
        SignalSpec("battery.plugged", "bool", SignalKind.FLAG, 0.0, 1.0),
        SignalSpec("net.recv_mbps", "megabyte_per_second", SignalKind.RATE, 0.0, 20000.0),
    ])


def make_sample(ts: float, **values: float | None) -> TelemetrySample:
    return TelemetrySample(
        ts=ts,
        readings={k: Reading(v) for k, v in values.items() if v is not None},
    )


@pytest.fixture
def scenario_samples():
    """A short deterministic scenario, cached per test session by parameters."""
    def _build(name: str = "idle_to_build", minutes: float = 60.0, seed: int = 7):
        spec = build_scenario(name, minutes=minutes, seed=seed)
        return generate(spec)
    return _build
