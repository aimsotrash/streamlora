"""Replay: determinism, acceleration, datasets, and the required scenarios."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from streamlora.config import Config
from streamlora.forecast.regime import RegimeClassifier
from streamlora.replay.player import ReplayPlayer
from streamlora.replay.recorder import export_from_db, import_to_db, read_dataset, write_dataset
from streamlora.replay.synthetic import SCENARIOS, build_scenario, generate, synthetic_registry
from streamlora.store.repo import Repos

from .conftest import T0


def _run(config, samples, registry, repos, run_id="r", **kw):
    p = ReplayPlayer(config, repos, registry, run_id=run_id, **kw)
    return p, p.run(samples)


# --------------------------------------------------------------------------
# scenario generation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_generates_deterministically(name):
    spec = build_scenario(name, minutes=30)
    a, reg = generate(spec)
    b, _ = generate(spec)
    assert len(a) == len(b) > 100
    for x, y in zip(a, b):
        assert x.ts == y.ts
        assert x.readings.keys() == y.readings.keys()
        for k in x.readings:
            assert x.readings[k].value == y.readings[k].value


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_values_stay_inside_declared_ranges(name):
    spec = build_scenario(name, minutes=30)
    samples, reg = generate(spec)
    for s in samples:
        for sig, r in s.readings.items():
            if r.value is None:
                continue
            spec_ = reg.get(sig)
            assert spec_ is not None
            if spec_.lo is not None:
                assert r.value >= spec_.lo - 1e-6, f"{sig}={r.value}"
            if spec_.hi is not None:
                assert r.value <= spec_.hi + 1e-6, f"{sig}={r.value}"


def test_unknown_scenario_is_a_clear_error():
    with pytest.raises(KeyError):
        build_scenario("no_such_scenario")


# -- the five scenarios the brief calls for ---------------------------------

def test_scenario_1_idle_to_sustained_workload():
    samples, _ = generate(build_scenario("idle_to_build", minutes=90))
    cpu = np.array([s.value("cpu.util_pct") for s in samples])
    third = len(cpu) // 3
    assert cpu[:third].mean() < 15.0
    assert cpu[third:2 * third].mean() > 55.0
    rc = RegimeClassifier()
    labels = [rc.update(s) for s in samples]
    assert "idle" in labels and "build" in labels


def test_scenario_2_discharge_then_charging():
    samples, _ = generate(build_scenario("discharge_then_charge", minutes=120))
    batt = np.array([s.value("battery.percent") for s in samples])
    plugged = np.array([s.value("battery.plugged") for s in samples])
    half = len(batt) // 2
    assert batt[:half].min() < batt[0]                 # it discharged
    assert batt[-1] > batt[half]                       # then charged
    assert plugged[10] == 0.0 and plugged[-10] == 1.0
    rc = RegimeClassifier()
    labels = {rc.update(s) for s in samples}
    assert "charging" in labels


def test_scenario_3_normal_workload_then_sudden_spike():
    samples, _ = generate(build_scenario("spike_burst", minutes=60))
    cpu = np.array([s.value("cpu.util_pct") for s in samples])
    assert cpu.max() > 85.0
    assert np.median(cpu) < 45.0
    # The spikes must be short relative to the run.
    assert (cpu > 80).mean() < 0.35


def test_scenario_4_behaviour_changes_permanently():
    samples, _ = generate(build_scenario("permanent_regime_change", minutes=120))
    gpu = np.array([s.value("gpu.util_pct") for s in samples])
    half = len(gpu) // 2
    assert gpu[:half].mean() < 10.0
    assert gpu[half + 60:].mean() > 60.0
    # and it does not revert
    assert gpu[-50:].mean() > 60.0


def test_scenario_5_dropouts_and_gaps_are_handled(config, repos):
    samples, reg = generate(build_scenario("gappy_idle", minutes=45))
    missing = sum(1 for s in samples if s.value("cpu.util_pct") is None)
    assert missing > 20
    repos.signals.register(reg.as_mapping().values())
    player, result = _run(config, samples, reg, repos, record_predictions=False)
    assert result.n_samples == len(samples)
    # Dropouts must be visible in the normaliser stats, not silently smoothed.
    assert player.normalizer.stats.stale + player.normalizer.stats.missing > 0


# --------------------------------------------------------------------------
# determinism of the whole pipeline
# --------------------------------------------------------------------------

def test_two_replays_of_the_same_dataset_are_identical(tmp_path):
    samples, reg = generate(build_scenario("idle_to_build", minutes=90))

    def once(tag):
        cfg = Config()
        cfg.general.data_dir = str(tmp_path / tag)
        cfg.forecast.horizons_s = [300.0]
        cfg.forecast.standardize_warmup = 40
        cfg.adapt.every_n_samples = 40
        cfg.adapt.gate_window = 30
        cfg.adapt.gate_min_samples = 15
        cfg.ensure_dirs()
        repos = Repos.open(cfg.db_file)
        repos.signals.register(reg.as_mapping().values())
        try:
            _p, res = _run(cfg, samples, reg, repos, run_id="d")
            preds = repos.predictions.resolved(run_id="d")
            sig = [(p.ts_target, p.model_kind, round(p.value, 9)) for p in preds]
            adapt = [(e["scope"], e["decision"], e["trigger"])
                     for e in repos.events.adapt_events(run_id="d")]
            return res.n_predictions, sig, adapt
        finally:
            repos.close()

    a = once("a")
    b = once("b")
    assert a[0] == b[0]
    assert a[1] == b[1], "predictions differ between identical replays"
    assert a[2] == b[2], "adaptation decisions differ between identical replays"


def test_replay_runs_far_faster_than_real_time(config, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=120))
    repos.signals.register(reg.as_mapping().values())
    _p, res = _run(config, samples, reg, repos, record_predictions=False)
    assert res.speedup > 50.0
    assert res.ts_end - res.ts_start == pytest.approx(120 * 60 - 5, abs=10)


def test_replay_uses_simulated_time_not_wall_clock(config, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=30))
    repos.signals.register(reg.as_mapping().values())
    player, _res = _run(config, samples, reg, repos)
    # Events must be stamped in the dataset's timeline, not today's.
    for ev in repos.events.adapt_events(run_id="r"):
        assert ev["ts"] <= samples[-1].ts + 1.0
    assert player.clock.now() == pytest.approx(samples[-1].ts)


def test_freeze_stops_learning_but_not_predicting(config, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=120))
    repos.signals.register(reg.as_mapping().values())
    boundary = samples[len(samples) // 2].ts
    player = ReplayPlayer(config, repos, reg, run_id="frozen", freeze_after_ts=boundary)
    res = player.run(samples)
    assert player._frozen
    assert player.engine.learned.frozen
    after = [p for p in repos.predictions.resolved(run_id="frozen", model_kind="rls")
             if p.ts_made > boundary]
    assert after, "a frozen model must still predict"
    for ev in repos.events.adapt_events(run_id="frozen"):
        assert ev["ts"] <= boundary + 1.0, "learning continued past the freeze"


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def test_dataset_round_trip_preserves_values_and_quality(tmp_path):
    samples, reg = generate(build_scenario("gappy_idle", minutes=20))
    path = str(tmp_path / "d.jsonl")
    info = write_dataset(path, samples, reg, notes="test")
    assert info.n_samples == len(samples)
    it, reg2, header = read_dataset(path)
    back = list(it)
    assert header["notes"] == "test"
    assert len(back) == len(samples)
    assert set(reg2.names()) == set(reg.names())
    for a, b in zip(samples, back):
        assert a.ts == b.ts
        for k in a.readings:
            assert a.readings[k].value == b.readings[k].value
            assert a.readings[k].quality == b.readings[k].quality


def test_dataset_gzip_round_trip(tmp_path):
    samples, reg = generate(build_scenario("idle_to_build", minutes=15))
    path = str(tmp_path / "d.jsonl.gz")
    write_dataset(path, samples, reg)
    back = list(read_dataset(path)[0])
    assert len(back) == len(samples)


def test_export_import_through_the_database(tmp_path, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=20))
    repos.signals.register(reg.as_mapping().values())
    repos.telemetry.insert_samples(samples, run_id="src")
    path = str(tmp_path / "e.jsonl")
    info = export_from_db(repos, path, run_id="src")
    assert info.n_samples == len(samples)
    other = Repos.open(str(tmp_path / "other.sqlite"))
    try:
        n = import_to_db(other, path, "dst")
        assert n == len(samples)
        assert other.telemetry.count(run_id="dst") == len(samples)
    finally:
        other.close()


def test_replays_are_identical_across_processes(tmp_path):
    """Determinism must survive a fresh interpreter.

    An in-process test cannot catch a seed derived from ``hash(str)``: PYTHONHASHSEED
    is fixed for the lifetime of one interpreter, so both replays agree with each
    other and disagree with tomorrow's run. This runs each replay in its own
    subprocess with *different* hash seeds, which is the condition that failed.
    """
    import json
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, %r)
        from streamlora.config import Config
        from streamlora.store.repo import Repos
        from streamlora.replay.player import ReplayPlayer
        from streamlora.replay.synthetic import build_scenario, generate
        from streamlora.util.logging import configure
        configure("error")
        out_dir = sys.argv[1]
        samples, reg = generate(build_scenario("idle_to_build", minutes=90))
        cfg = Config()
        cfg.general.data_dir = out_dir
        cfg.forecast.horizons_s = [300.0]
        cfg.forecast.standardize_warmup = 40
        cfg.adapt.every_n_samples = 40
        cfg.adapt.gate_window = 30
        cfg.adapt.gate_min_samples = 15
        cfg.ensure_dirs()
        repos = Repos.open(cfg.db_file)
        repos.signals.register(reg.as_mapping().values())
        ReplayPlayer(cfg, repos, reg, run_id="d").run(samples)
        preds = repos.predictions.resolved(run_id="d", model_kind="rls")
        print(json.dumps([[p.ts_target, round(p.value, 9)] for p in preds]))
        repos.close()
        """
    ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),)
    path = tmp_path / "run.py"
    path.write_text(script)

    outs = []
    for i, seed in enumerate(("0", "12345")):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, str(path), str(tmp_path / f"proc{i}")],
            capture_output=True, text=True, env=env, timeout=900,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0], "replay produced no learned predictions"
    assert outs[0] == outs[1], (
        "replays in separate processes disagree; something is seeded from a "
        "process-dependent value"
    )
