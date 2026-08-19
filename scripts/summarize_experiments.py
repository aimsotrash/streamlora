"""Collapse experiment reports into the tables used in docs/experiments.md.

Reads the JSON reports written by ``streamlora experiment`` so that every number
in the documentation is traceable to a stored run rather than transcribed by
hand.
"""
from __future__ import annotations

import json
import os
import sys


def load(runs_dir: str) -> list[dict]:
    out = []
    for fn in sorted(os.listdir(runs_dir)):
        if not fn.endswith(".json") or fn == "latest.json":
            continue
        with open(os.path.join(runs_dir, fn)) as fh:
            d = json.load(fh)
        if "arms" in d and "dataset" in d:
            out.append(d)
    return out


BASELINES = ("persistence", "ewma", "moving_average", "linear_trend")


def main(runs_dir: str) -> int:
    reports = load(runs_dir)
    if not reports:
        print(f"no experiment reports in {runs_dir}")
        return 1
    for rep in reports:
        first = rep["arms"][0]
        scopes = sorted(
            {(m["signal"], m["horizon_s"]) for m in first["metrics"]},
            key=lambda k: (k[0], k[1]),
        )
        arm_names = [a["arm"]["name"] for a in rep["arms"]]
        print(f"\n### {rep['name']}   ({rep['dataset']}, train_frac {rep['train_frac']})")
        head = (f"| scope | best baseline | " + " | ".join(arm_names) +
                " | best arm vs baseline |")
        print(head)
        print("|" + "---|" * (len(arm_names) + 3))
        for sig, hor in scopes:
            bl = [m for m in first["metrics"]
                  if m["signal"] == sig and m["horizon_s"] == hor and m["arm"] in BASELINES]
            bl = [m for m in bl if m["mae"] is not None]
            if not bl:
                continue
            best_bl = min(bl, key=lambda m: m["mae"])
            if best_bl["mae"] < 1e-6:
                continue      # degenerate scope (constant signal)
            cells = []
            best_arm_mae = None
            counts: list[int] = []
            for a in rep["arms"]:
                m = next((x for x in a["metrics"]
                          if x["arm"] == "rls" and x["signal"] == sig
                          and x["horizon_s"] == hor), None)
                if m and m["mae"] is not None:
                    cells.append(f"{m['mae']:.3f}")
                    counts.append(int(m["n"]))
                    if best_arm_mae is None or m["mae"] < best_arm_mae:
                        best_arm_mae = m["mae"]
                else:
                    cells.append("–")
            skill = (1 - best_arm_mae / best_bl["mae"]) if best_arm_mae else None
            uneven = bool(counts and min(counts) > 0 and max(counts) / min(counts) > 2.0)
            tail = (f" | {skill:+.1%}{' (uneven n)' if uneven else ''} |"
                    if skill is not None else " | - |")
            print(f"| {sig}@{int(hor)} | {best_bl['mae']:.3f} ({best_bl['arm']}) | "
                  + " | ".join(cells) + tail)
        # The headline comparison: does continual adaptation beat a static model
        # trained on the same prefix and then frozen?
        static = next((a for a in rep["arms"] if a["arm"]["name"] == "static"), None)
        online = [a for a in rep["arms"] if a["arm"]["name"].startswith("online")]
        if static and online:
            print()
            print("| scope | static | best online | online vs static |")
            print("|---|---|---|---|")
            wins = losses = 0
            for sig, hor in scopes:
                sm = next((m for m in static["metrics"] if m["arm"] == "rls"
                           and m["signal"] == sig and m["horizon_s"] == hor), None)
                if not sm or sm["mae"] is None or sm["mae"] < 1e-9:
                    continue
                cands = []
                for a in online:
                    m = next((x for x in a["metrics"] if x["arm"] == "rls"
                              and x["signal"] == sig and x["horizon_s"] == hor), None)
                    if m and m["mae"] is not None:
                        cands.append(m["mae"])
                if not cands:
                    continue
                best = min(cands)
                # Only compare arms scored on comparable sample counts.
                ns = [sm["n"]] + [
                    m["n"] for a in online
                    for m in a["metrics"]
                    if m["arm"] == "rls" and m["signal"] == sig and m["horizon_s"] == hor
                    and m["mae"] is not None
                ]
                if min(ns) > 0 and max(ns) / min(ns) > 2.0:
                    print(f"| {sig}@{int(hor)} | {sm['mae']:.3f} | {best:.3f} | "
                          f"n/a (uneven n: {min(ns)}-{max(ns)}) |")
                    continue
                rel = 1 - best / sm["mae"]
                wins += rel > 0.01
                losses += rel < -0.01
                print(f"| {sig}@{int(hor)} | {sm['mae']:.3f} | {best:.3f} | {rel:+.1%} |")
            print(f"\nonline better on {wins} scopes, worse on {losses}")
        print()
        print("| arm | promoted | rejected | rolled back | drift events |")
        print("|---|---|---|---|---|")
        for a in rep["arms"]:
            ad = a.get("adapt") or {}
            dr = a.get("drift") or {}
            print(f"| {a['arm']['name']} | {ad.get('promoted', '–')} | "
                  f"{ad.get('rejected', '–')} | {ad.get('rolled_back', '–')} | "
                  f"{dr.get('events', '–')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/exp_synthetic/runs"))
