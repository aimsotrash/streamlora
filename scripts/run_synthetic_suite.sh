#!/usr/bin/env bash
# Core ablation across every synthetic scenario. Used to produce the numbers in
# docs/experiments.md. Each scenario is a separate experiment because the
# scenarios are not comparable to each other -- only arms within one are.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-data/exp_synthetic}
MIN=${2:-300}
SUITE=${3:-core}
PY=${PY:-.venv/bin/python}
for scen in permanent_regime_change idle_to_build spike_burst discharge_then_charge gappy_idle; do
  echo "=============================================================="
  echo "scenario: $scen   suite: $SUITE   minutes: $MIN"
  echo "=============================================================="
  $PY -m streamlora experiment \
    --data-dir "$DATA" --log-level error \
    --dataset "scenario:$scen" --minutes "$MIN" --suite "$SUITE" \
    --train-frac 0.5 --name "${SUITE}-${scen}"
done
