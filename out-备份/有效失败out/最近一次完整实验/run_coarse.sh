#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT="$ROOT/out/second_memory_pressure_20260912"
cd "$ROOT" || exit 2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"; printf "FINISHED %s exit=%s\n" "$(date -Is)" "$rc"' EXIT
printf 'START %s root=%s\n' "$(date -Is)" "$ROOT"
if ! conda run -n tailguardkv-base python "$OUT/preflight.py" "$ROOT" "$OUT" > "$OUT/logs/preflight.log" 2>&1; then
  cat "$OUT/logs/preflight.log" >&2
  echo 'PREFLIGHT_FAILED: no experiment cells were started' >&2
  exit 2
fi
cat "$OUT/logs/preflight.log"
MEASUREMENTS="$ROOT/out/pilot_tight_budget_coarse_batch007_20260911/batch_outputs/batch007/baseline_wide_sweep/profile_tables/pilot_smoke_measured_profiles_batch007.csv"
printf 'budget_mib,seed,process_return_code,policy_csv,log_path\n' > "$OUT/run_status.csv"
failures=0
for budget in 50 35 30 25 20 15; do
  csv="$OUT/policy_tables/pilot_smoke_measured_policy_eps0p1_delta0p1_mem${budget}.csv"
  log="$OUT/logs/mem${budget}.log"
  printf 'CELL_START %s budget=%s\n' "$(date -Is)" "$budget" | tee "$log"
  conda run -n tailguardkv-base python -m run_util.run_policies \
    --config "$OUT/coarse_config.yaml" \
    --backend measured_replay --measurements "$MEASUREMENTS" \
    --output "$csv" --epsilon 0.1 --delta 0.1 \
    --memory-budget-mib "$budget" >> "$log" 2>&1
  rc=$?
  printf 'CELL_END %s budget=%s return_code=%s\n' "$(date -Is)" "$budget" "$rc" | tee -a "$log"
  printf '%s,%s,%s,%s,%s\n' "$budget" 20260906 "$rc" "$csv" "$log" >> "$OUT/run_status.csv"
  if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
done
if [ "$failures" -eq 0 ]; then
  conda run -n tailguardkv-base python "$ROOT/scripts/aggregate_baseline_wide_sweep.py" \
    --input-dir "$OUT/policy_tables" --output "$OUT/policy_tables/coarse_total_summary.csv" \
    > "$OUT/logs/aggregate.log" 2>&1
  aggregate_rc=$?
  printf 'AGGREGATION status=%s\n' "$aggregate_rc"
  if [ "$aggregate_rc" -ne 0 ]; then
    cat "$OUT/logs/aggregate.log" >&2
    exit "$aggregate_rc"
  fi
else
  echo "AGGREGATION_SKIPPED failed_budgets=$failures; preserve per-budget errors and any request CSVs" >&2
  exit 1
fi
printf 'COARSE_DONE %s budget_runs=6 expected_cells=30 failed_budgets=0\n' "$(date -Is)"
