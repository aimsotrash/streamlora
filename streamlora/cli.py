"""Command-line interface.

Subcommands map one-to-one onto the pipeline stages, so the system can be
exercised piecewise:

    streamlora doctor        what telemetry does this machine actually expose?
    streamlora collect       run the live collector
    streamlora export        snapshot recorded telemetry to a portable dataset
    streamlora replay        drive a dataset through the full pipeline
    streamlora eval          metrics from stored predictions
    streamlora experiment    run an ablation suite and write a report
    streamlora scenarios     list synthetic scenarios
    streamlora models        inspect the version registry
    streamlora lora          train / evaluate the language adaptation layer
    streamlora serve         API + dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Sequence

from .config import Config
from .util.ids import config_hash, short_uid
from .util.logging import configure, get_logger

log = get_logger("cli")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "data_dir", None):
        cfg.general.data_dir = args.data_dir
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, _, v = item.partition("=")
        try:
            cfg.override(k.strip(), v.strip())
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"bad --set {item!r}: {exc}") from exc
    if getattr(args, "log_level", None):
        cfg.general.log_level = args.log_level
    if getattr(args, "log_json", False):
        cfg.general.log_format = "json"
    configure(cfg.general.log_level, cfg.general.log_format, cfg.general.log_file)
    cfg.ensure_dirs()
    return cfg


def _emit(obj: object, as_json: bool, text: str | None = None) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    """Probe the machine and report what telemetry is available."""
    cfg = _load_config(args)
    from .telemetry.registry import SourceRegistry, default_sources
    from .util.clock import RealClock

    reg = SourceRegistry(default_sources(None), RealClock())
    health = reg.probe_all()
    sample_raw, errors = reg.read_all(time.time())
    payload = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "data_dir": os.path.abspath(cfg.general.data_dir),
        "db": cfg.db_file,
        "sources": {n: h.as_dict() for n, h in health.items()},
        "signals": reg.expected_signals(),
        "n_signals": len(reg.signals),
        "read_errors": errors,
        "optional_extras": _extras_status(),
    }
    reg.close()
    if args.json:
        _emit(payload, True)
        return 0
    lines = [
        f"StreamLoRA doctor",
        f"  python {payload['python']} on {payload['platform']}",
        f"  data dir: {payload['data_dir']}",
        f"  signals available: {payload['n_signals']}",
        "",
        "sources:",
    ]
    for n, h in sorted(health.items()):
        mark = "ok  " if h.available else "MISS"
        lines.append(f"  [{mark}] {n:12s} {len(h.provides):2d} signals  {h.detail}")
    lines.append("")
    lines.append("optional extras:")
    for k, v in payload["optional_extras"].items():
        lines.append(f"  [{'ok  ' if v['available'] else 'MISS'}] {k:14s} {v['detail']}")
    missing = [s for s in cfg.forecast.targets if s not in reg.signals]
    if missing:
        lines.append("")
        lines.append(f"WARNING: configured forecast targets unavailable here: {missing}")
        lines.append("         those targets will be skipped; everything else still runs.")
    lines.append("")
    lines.append("current reading sample:")
    for k in sorted(sample_raw)[:12]:
        v = sample_raw[k]
        lines.append(f"  {k:26s} {'n/a' if v is None else f'{v:.3f}'}")
    if len(sample_raw) > 12:
        lines.append(f"  ... and {len(sample_raw) - 12} more")
    print("\n".join(lines))
    return 0


def _extras_status() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    try:
        import pynvml  # noqa: F401
        out["nvidia-ml-py"] = {"available": True, "detail": "fast NVML GPU reads"}
    except Exception:
        out["nvidia-ml-py"] = {
            "available": False, "detail": "not installed; GPU falls back to nvidia-smi (45 ms/read)"
        }
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers
        dev = "cpu"
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        out["lora (torch/peft)"] = {
            "available": True,
            "detail": f"transformers {transformers.__version__}, device={dev}",
        }
    except Exception:
        out["lora (torch/peft)"] = {
            "available": False,
            "detail": "not installed; language layer runs the template backend",
        }
    return out


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .adapt.controller import AdaptationController
    from .adapt.registry import ModelVersionRegistry
    from .forecast.engine import ForecastEngine
    from .pipeline import build_telemetry_pipeline

    run_id = args.run_id or f"live-{time.strftime('%Y%m%d-%H%M%S')}"
    pipe = build_telemetry_pipeline(cfg, run_id=run_id, origin="live")
    pipe.repos.runs.start(
        run_id, "collect", time.time(), cfg.to_dict(), config_hash(cfg.to_dict()),
        notes=args.notes or "",
    )
    engine = None
    if args.forecast:
        engine = ForecastEngine(
            cfg, pipe.repos, pipe.registry.signals, run_id=run_id,
            enable_drift=cfg.drift.enabled, enable_adapt=cfg.adapt.enabled,
        )
        if cfg.adapt.enabled:
            engine.adapt = AdaptationController(
                forecaster=engine.learned,
                registry=ModelVersionRegistry(
                    pipe.repos, cfg.models_dir, keep=cfg.adapt.keep_versions
                ),
                repos=pipe.repos, config=cfg.adapt, run_id=run_id,
                seed=cfg.general.seed, config_fingerprint=config_hash(cfg.to_dict()),
            )
        pipe.collector.subscribe(engine.on_sample)

    stopping = {"v": False}

    def handle(signum: int, _frame: object) -> None:
        stopping["v"] = True
        pipe.collector.stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    duration = args.minutes * 60.0 if args.minutes else None
    log.info(
        "collect starting", run_id=run_id, interval_s=cfg.collect.interval_s,
        forecast=bool(args.forecast), minutes=args.minutes,
    )
    try:
        pipe.collector.run(duration_s=duration)
    finally:
        n = pipe.repos.telemetry.count(run_id=run_id)
        summary: dict[str, object] = {"samples": n, **pipe.collector.stats.as_dict()}
        if engine is not None:
            summary["engine"] = engine.stats.as_dict()
        pipe.repos.runs.finish(
            run_id, time.time(), "stopped" if stopping["v"] else "complete", summary=summary
        )
        log.info("collect finished", run_id=run_id, samples=n)
        print(json.dumps({"run_id": run_id, "samples": n}, indent=2))
        pipe.close()
    return 0


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .replay.recorder import export_from_db
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        info = export_from_db(
            repos, args.out, run_id=args.run_id, ts_from=args.ts_from, ts_to=args.ts_to,
            notes=args.notes or "",
        )
    finally:
        repos.close()
    _emit(info.as_dict(), args.json,
          f"wrote {info.n_samples} samples spanning "
          f"{info.duration_s / 3600:.2f} h to {info.path}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .replay.recorder import import_to_db
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    run_id = args.run_id or f"import-{short_uid()}"
    try:
        n = import_to_db(repos, args.path, run_id)
    finally:
        repos.close()
    _emit({"run_id": run_id, "samples": n}, args.json,
          f"imported {n} samples as run_id={run_id}")
    return 0


# ---------------------------------------------------------------------------
# scenarios / replay
# ---------------------------------------------------------------------------

def cmd_scenarios(args: argparse.Namespace) -> int:
    from .replay.synthetic import SCENARIOS, build_scenario

    out = []
    for name in sorted(SCENARIOS):
        spec = build_scenario(name)
        out.append({
            "name": name, "description": spec.description,
            "default_minutes": round(spec.duration_s / 60.0, 1),
            "segments": [s.name for s in spec.segments],
        })
    if args.json:
        _emit(out, True)
        return 0
    for s in out:
        print(f"{s['name']:26s} {s['default_minutes']:6.0f} min  {s['description']}")
        print(f"{'':26s} segments: {' -> '.join(s['segments'])}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .experiments.runner import DatasetRef
    from .replay.player import ReplayPlayer
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        ref = DatasetRef.parse(args.dataset, minutes=args.minutes, seed=args.seed)
        samples, registry = ref.load(repos)
        if not samples:
            raise SystemExit(f"dataset {ref.label()} produced no samples")
        repos.signals.register(registry.as_mapping().values())
        run_id = args.run_id or f"replay-{time.strftime('%Y%m%d-%H%M%S')}"
        repos.runs.start(
            run_id, "replay", time.time(), cfg.to_dict(), config_hash(cfg.to_dict()),
            notes=f"dataset={ref.label()}",
        )
        player = ReplayPlayer(
            cfg, repos, registry, run_id=run_id, speed=args.speed,
            store_telemetry=args.store_telemetry, record_predictions=True,
            enable_drift=cfg.drift.enabled, enable_adapt=cfg.adapt.enabled,
        )
        result = player.run(samples, limit=args.limit)
        repos.runs.finish(run_id, time.time(), "complete", summary=result.as_dict())
        payload = result.as_dict()
        payload["report"] = result.engine_report
        if args.json:
            _emit(payload, True)
        else:
            r = result
            print(f"run_id           {r.run_id}")
            print(f"samples          {r.n_samples}")
            print(f"predictions      {r.n_predictions}")
            print(f"resolved         {r.n_resolved}")
            print(f"simulated        {(r.ts_end - r.ts_start) / 3600:.2f} h")
            print(f"wall             {r.wall_s:.2f} s  ({r.speedup:.0f}x real time)")
            ad = (r.engine_report or {}).get("adapt") or {}
            if ad:
                print(f"adaptation       promoted={ad.get('promoted')} "
                      f"rejected={ad.get('rejected')} rolled_back={ad.get('rolled_back')}")
            dr = (r.engine_report or {}).get("drift") or {}
            if dr:
                print(f"drift            events={dr.get('events')} "
                      f"suppressed={dr.get('suppressed')}")
            print()
            print("Run `streamlora eval --run-id " + r.run_id + "` for metrics.")
    finally:
        repos.close()
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .evaluate import report as R
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        recs = repos.predictions.resolved(
            run_id=args.run_id, ts_from=args.ts_from, ts_to=args.ts_to
        )
        if args.since_minutes:
            cutoff = time.time() - args.since_minutes * 60.0
            recs = [r for r in recs if r.ts_target >= cutoff]
        if not recs:
            print("no resolved predictions found for that selection", file=sys.stderr)
            return 1
        rows = R.summarize(recs, reference=args.reference, by_regime=args.by_regime)
        if args.over_time:
            rows = R.summarize_over_time(recs, n_windows=args.over_time)
        if args.json:
            _emit([r.as_dict() for r in rows], True)
            return 0
        print(R.to_table(rows))
        print()
        print("skill = 1 - MAE / MAE_" + args.reference + "   (positive is better)")
        hdr = f"{'arm':<22}{'scope':<26}{'MAE':>10}{'RMSE':>10}{'skill':>9}{'cover':>8}{'DM p':>9}{'n':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(rows, key=lambda x: (x.signal, x.horizon_s, x.arm)):
            lab = r.scope + (f" [{r.regime}]" if r.regime else "") + (f" ({r.window})" if r.window else "")
            print(
                f"{r.arm:<22}{lab:<26}"
                f"{(r.m.mae if r.m.mae is not None else float('nan')):>10.4f}"
                f"{(r.m.rmse if r.m.rmse is not None else float('nan')):>10.4f}"
                f"{('' if r.m.skill is None else f'{r.m.skill:+.1%}'):>9}"
                f"{('' if r.m.coverage is None else f'{r.m.coverage:.2f}'):>8}"
                f"{('' if r.dm_p is None else f'{r.dm_p:.4f}'):>9}"
                f"{r.m.n:>7}"
            )
    finally:
        repos.close()
    return 0


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def cmd_experiment(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .experiments.ablations import ALL_ARMS, SUITES
    from .experiments.runner import DatasetRef, ExperimentRunner

    if args.arms:
        names = [a.strip() for a in args.arms.split(",")]
        unknown = [n for n in names if n not in ALL_ARMS]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; have {sorted(ALL_ARMS)}")
        arms = [ALL_ARMS[n] for n in names]
    else:
        if args.suite not in SUITES:
            raise SystemExit(f"unknown suite {args.suite!r}; have {sorted(SUITES)}")
        arms = SUITES[args.suite]

    ref = DatasetRef.parse(args.dataset, minutes=args.minutes, seed=args.seed)
    runner = ExperimentRunner(cfg)
    try:
        result = runner.run(
            name=args.name or f"{args.suite}-{ref.name}", dataset=ref, arms=arms,
            train_frac=args.train_frac, notes=args.notes or "",
            progress=None if args.json else (lambda m: print(f"... {m}", file=sys.stderr)),
        )
    finally:
        runner.close()
    if args.json:
        _emit(result.as_dict(), True)
        return 0
    print()
    print(f"experiment  {result.name}   id={result.experiment_id}")
    print(f"dataset     {result.dataset}")
    print(f"train_frac  {result.train_frac}   (metrics computed on the held-out tail only)")
    print()
    print("MAE by scope and arm (learned model per arm; baselines shared)")
    print(result.comparison_table())
    print()
    print(f"{'arm':<24}{'promoted':>10}{'rejected':>10}{'rolled_back':>13}{'drift':>8}")
    print("-" * 65)
    for a in result.arms:
        ad = (a.replay.engine_report or {}).get("adapt") or {}
        dr = (a.replay.engine_report or {}).get("drift") or {}
        print(f"{a.arm.name:<24}{ad.get('promoted', 0):>10}{ad.get('rejected', 0):>10}"
              f"{ad.get('rolled_back', 0):>13}{dr.get('events', 0):>8}")
    print()
    print(f"report: {os.path.join(cfg.runs_dir, result.experiment_id + '.json')}")
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def cmd_models(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        hist = repos.models.history(kind=args.kind, scope=args.scope, limit=args.limit)
        if args.json:
            _emit(hist, True)
            return 0
        if not hist:
            print("no model versions registered yet")
            return 0
        hdr = f"{'kind':<16}{'scope':<26}{'version':<22}{'act':>4}{'n_train':>9}{'created':>21}"
        print(hdr)
        print("-" * len(hdr))
        for r in hist:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_ts"]))
            print(f"{r['kind']:<16}{r['scope']:<26}{r['version']:<22}"
                  f"{'*' if r['active'] else '':>4}{r['n_train']:>9}{ts:>21}")
        return 0
    finally:
        repos.close()


def cmd_rollback(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .adapt.registry import ModelVersionRegistry
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        reg = ModelVersionRegistry(repos, cfg.models_dir, keep=cfg.adapt.keep_versions)
        cur = reg.active(args.scope)
        prev = reg.rollback(args.scope, time.time())
        if prev is None:
            print(f"no earlier version available for scope {args.scope!r}", file=sys.stderr)
            return 1
        repos.events.add_adapt(
            ts=time.time(), scope=args.scope, kind="forecast", trigger="manual",
            decision="rolled_back", active_version=cur.version if cur else None,
            candidate_version=prev.version, metric_name="mae", metric_before=None,
            metric_after=None, gate_n=0, n_train=0, duration_ms=0.0,
            detail={"reason": "manual rollback via CLI"}, run_id=None,
        )
        _emit({"scope": args.scope, "from": cur.version if cur else None, "to": prev.version},
              args.json,
              f"rolled back {args.scope}: {cur.version if cur else '?'} -> {prev.version}")
        return 0
    finally:
        repos.close()


# ---------------------------------------------------------------------------
# lora / language
# ---------------------------------------------------------------------------

def cmd_lora(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .language.experiment import run_language_experiment

    return run_language_experiment(cfg, args)


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .language.service import LanguageService
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        svc = LanguageService(cfg, repos)
        answer = svc.ask(args.question, now=time.time())
        if args.json:
            _emit(answer.as_dict(), True)
        else:
            print(answer.text)
            if answer.evidence:
                print()
                print("evidence:")
                for e in answer.evidence:
                    print(f"  - {e}")
            print()
            print(f"[backend={answer.backend} adapter={answer.adapter or 'none'} "
                  f"grounded={answer.grounded} ms={answer.latency_ms:.0f}]")
    finally:
        repos.close()
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .language.feedback import FeedbackStore
    from .store.repo import Repos

    repos = Repos.open(cfg.db_file)
    try:
        store = FeedbackStore(repos)
        fid = store.add(
            kind=args.kind, text=args.text or "", label=args.label,
            prediction_id=args.prediction_id, signal=args.signal, now=time.time(),
        )
        _emit({"id": fid}, args.json, f"recorded feedback id={fid}")
    finally:
        repos.close()
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .api.app import serve

    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    return serve(cfg, host=host, port=port, collect=not args.no_collect, reload=False)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_global_options(p: argparse.ArgumentParser, suppress: bool = False) -> None:
    """Attach the options accepted both before and after the subcommand.

    ``suppress`` must be True for the shared parent parser. argparse merges the
    subparser's results into the *same* namespace, so an option present in both
    parsers and left unset after the subcommand overwrites the value given
    before it with its default. With ``SUPPRESS`` the attribute is simply absent
    when unset, so the earlier value survives. Without this,
    ``streamlora --data-dir X collect`` silently writes to the default directory.
    """
    d: dict[str, Any] = {"default": argparse.SUPPRESS} if suppress else {}
    p.add_argument("--config", help="TOML or JSON config file", **d)
    p.add_argument("--data-dir", help="override general.data_dir", **d)
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config value, e.g. --set forecast.horizons_s=60,300", **d)
    p.add_argument("--log-level", choices=["debug", "info", "warning", "error"], **d)
    p.add_argument("--log-json", action="store_true", help="structured JSON logs", **d)
    p.add_argument("--json", action="store_true", help="machine-readable output", **d)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="streamlora",
        description="Continuous learning and forecasting for personal computer telemetry.",
    )
    _add_global_options(p)
    # The same options are attached to every subcommand via a parent parser, so
    # both `streamlora --set k=v replay ...` and `streamlora replay --set k=v ...`
    # work. Only accepting the first form is a papercut that costs real time
    # when the flag is the thing being iterated on.
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress=True)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=argparse.ArgumentParser)

    d = sub.add_parser("doctor", parents=[common], help="probe available telemetry and extras")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("collect", parents=[common], help="run the live telemetry collector")
    c.add_argument("--minutes", type=float, help="stop after this many minutes")
    c.add_argument("--run-id")
    c.add_argument("--notes")
    c.add_argument("--forecast", action="store_true",
                   help="also run the forecasting/adaptation pipeline live")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("export", parents=[common], help="snapshot telemetry to a dataset file")
    e.add_argument("out")
    e.add_argument("--run-id")
    e.add_argument("--ts-from", type=float)
    e.add_argument("--ts-to", type=float)
    e.add_argument("--notes")
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("import", parents=[common], help="load a dataset file into the database")
    i.add_argument("path")
    i.add_argument("--run-id")
    i.set_defaults(func=cmd_import)

    s = sub.add_parser("scenarios", parents=[common], help="list synthetic scenarios")
    s.set_defaults(func=cmd_scenarios)

    r = sub.add_parser("replay", parents=[common], help="drive a dataset through the full pipeline")
    r.add_argument("dataset", help="scenario:<name> | file:<path> | db:<run_id>")
    r.add_argument("--minutes", type=float, help="scenario length")
    r.add_argument("--seed", type=int)
    r.add_argument("--run-id")
    r.add_argument("--limit", type=int)
    r.add_argument("--speed", type=float, default=0.0,
                   help="0 = as fast as possible; N = N x real time")
    r.add_argument("--store-telemetry", action="store_true")
    r.set_defaults(func=cmd_replay)

    v = sub.add_parser("eval", parents=[common], help="metrics from stored predictions")
    v.add_argument("--run-id")
    v.add_argument("--ts-from", type=float)
    v.add_argument("--ts-to", type=float)
    v.add_argument("--since-minutes", type=float)
    v.add_argument("--reference", default="persistence")
    v.add_argument("--by-regime", action="store_true")
    v.add_argument("--over-time", type=int, metavar="N",
                   help="split into N chronological windows")
    v.set_defaults(func=cmd_eval)

    x = sub.add_parser("experiment", parents=[common], help="run an ablation suite")
    x.add_argument("--dataset", required=True, help="scenario:<name> | file:<path> | db:<run_id>")
    x.add_argument("--suite", default="core")
    x.add_argument("--arms", help="comma-separated arm names, overrides --suite")
    x.add_argument("--minutes", type=float)
    x.add_argument("--seed", type=int)
    x.add_argument("--train-frac", type=float, default=0.5)
    x.add_argument("--name")
    x.add_argument("--notes")
    x.set_defaults(func=cmd_experiment)

    m = sub.add_parser("models", parents=[common], help="inspect the version registry")
    m.add_argument("--kind")
    m.add_argument("--scope")
    m.add_argument("--limit", type=int, default=50)
    m.set_defaults(func=cmd_models)

    rb = sub.add_parser("rollback", parents=[common], help="activate the previous version of a scope")
    rb.add_argument("scope", help="e.g. cpu.util_pct@300")
    rb.set_defaults(func=cmd_rollback)

    lo = sub.add_parser("lora", parents=[common], help="train / evaluate the language adaptation layer")
    lo.add_argument("action", choices=["build-data", "train", "eval", "compare", "status"])
    lo.add_argument("--examples", type=int, default=0, help="cap on generated examples")
    lo.add_argument("--out")
    lo.add_argument("--arms", help="comma-separated language arms for compare")
    lo.add_argument("--persona", help="path to a persona spec JSON")
    lo.set_defaults(func=cmd_lora)

    a = sub.add_parser("ask", parents=[common], help="ask a grounded question about your telemetry")
    a.add_argument("question")
    a.set_defaults(func=cmd_ask)

    fb = sub.add_parser("feedback", parents=[common], help="record structured feedback")
    fb.add_argument("kind", choices=["forecast_useful", "event_happened", "label", "note"])
    fb.add_argument("--text")
    fb.add_argument("--label")
    fb.add_argument("--prediction-id", type=int)
    fb.add_argument("--signal")
    fb.set_defaults(func=cmd_feedback)

    sv = sub.add_parser("serve", parents=[common], help="run the API and dashboard")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.add_argument("--no-collect", action="store_true",
                    help="serve stored data only; do not start the collector")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("command failed", command=getattr(args, "cmd", "?"))
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
