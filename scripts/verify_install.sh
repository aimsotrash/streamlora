#!/usr/bin/env bash
# End-to-end smoke test of a fresh checkout. Run after cloning to confirm the
# whole pipeline works on this machine before trusting any numbers from it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== 1/6  what can this machine measure?"
$PY -m streamlora doctor --data-dir "$TMP" | sed -n '1,12p'

echo
echo "== 2/6  collect 30 seconds of real telemetry"
$PY -m streamlora collect --data-dir "$TMP" --minutes 0.5 --run-id smoke

echo
echo "== 3/6  replay a synthetic scenario through the full pipeline"
$PY -m streamlora replay scenario:idle_to_build --data-dir "$TMP" \
  --minutes 90 --run-id smoke-replay --log-level error

echo
echo "== 4/6  metrics against the baselines"
$PY -m streamlora eval --data-dir "$TMP" --run-id smoke-replay | head -20

echo
echo "== 5/6  a grounded answer with no language model"
$PY -m streamlora ask "What is my machine doing right now?" --data-dir "$TMP"

echo
echo "== 6/6  language layer status"
$PY -m streamlora lora status --data-dir "$TMP" | head -8

echo
echo "all six stages completed"
