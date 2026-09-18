#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="$root/out/full_25session_baseline_9_14"
mkdir -p "$run_root/logs"
trap 'result=$?; printf "%s\n" "$result" > "$run_root/full.exit"' EXIT

if [[ ! -f "$run_root/preflight_attempt3.exit" || "$(<"$run_root/preflight_attempt3.exit")" != 0 ]]; then
  echo "two-session preflight has not passed; refusing full run" >&2
  exit 2
fi

cd "$root"
conda run --no-capture-output -n tailguardkv-base python -m pytest \
  tests/test_session25_fourth_preflight.py tests/test_session25_fourth_sweeps.py \
  tests/test_session_aggregation.py tests/test_diagnostic_batch_runner.py -q

conda run --no-capture-output -n tailguardkv-base python \
  scripts/run_diagnostic_session_batches.py \
  --fixture data/fixtures/diagnostic_session27_exclude_batch7.jsonl \
  --config configs/pilot_diagnostic_session25_fourth.yaml \
  --run-root "$run_root/full" --sessions-per-batch 2 --profile-only

conda run --no-capture-output -n tailguardkv-base python \
  scripts/run_session25_fourth_sweeps.py \
  --config configs/pilot_diagnostic_session25_fourth.yaml \
  --run-root "$run_root/full"
