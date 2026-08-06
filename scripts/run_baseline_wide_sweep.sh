#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/baseline_wide_sweep.yaml"
MEASUREMENTS_PATH="$ROOT_DIR/out/20260806_122619_pilot/profile_tables/pilot_smoke_measured_profiles.csv"
POLICY_DIR="$ROOT_DIR/out/baseline_wide_sweep/policy_tables"
LOG_DIR="$ROOT_DIR/out/baseline_wide_sweep/logs"
SUMMARY_PATH="$POLICY_DIR/baseline_wide_sweep_total_summary.csv"
CONDA_ENV="tailguardkv-base"
BUDGETS=()
EPSILONS=()
DELTAS=()

mkdir -p "$POLICY_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=0,1

if [[ ! -f "$MEASUREMENTS_PATH" ]]; then
  echo "MISSING_MEASUREMENTS: $MEASUREMENTS_PATH" >&2
  echo "先异步重建 profile，再重跑本脚本:" >&2
  echo "CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n $CONDA_ENV python -m run_util.build_profile_table --config $CONFIG_PATH --output $MEASUREMENTS_PATH --no-dry-run > $LOG_DIR/profile_rebuild.nohup.log 2>&1 < /dev/null &" >&2
  exit 2
fi

if ! conda run -n "$CONDA_ENV" python -c "from pathlib import Path; from run_util.experiment_common import read_measurements, validate_profile_measurements; measurements = read_measurements(Path(r'$MEASUREMENTS_PATH')); validate_profile_measurements(measurements, r'$MEASUREMENTS_PATH', require_measured=True)" >/dev/null 2>&1
then
  echo "UNUSABLE_MEASUREMENTS: $MEASUREMENTS_PATH 不是 measured profile 表" >&2
  echo "先异步重建 profile，再重跑本脚本:" >&2
  echo "CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n $CONDA_ENV python -m run_util.build_profile_table --config $CONFIG_PATH --output $MEASUREMENTS_PATH --no-dry-run > $LOG_DIR/profile_rebuild.nohup.log 2>&1 < /dev/null &" >&2
  exit 2
fi

slug_number() {
  local value="$1"
  if [[ "$value" == -* ]]; then
    value="m${value#-}"
  fi
  echo "${value//./p}"
}

readarray -t GRID_VALUES < <(
  conda run -n "$CONDA_ENV" python "$ROOT_DIR/scripts/baseline_wide_sweep_grid.py" --config "$CONFIG_PATH" |
    python -c 'import json, sys; grid = json.load(sys.stdin); [print(key + "=" + ",".join(f"{value:g}" for value in grid[key])) for key in ("memory_budgets_mib", "epsilons", "deltas")]'
)

for line in "${GRID_VALUES[@]}"; do
  case "$line" in
    memory_budgets_mib=*)
      IFS=',' read -r -a BUDGETS <<< "${line#*=}"
      ;;
    epsilons=*)
      IFS=',' read -r -a EPSILONS <<< "${line#*=}"
      ;;
    deltas=*)
      IFS=',' read -r -a DELTAS <<< "${line#*=}"
      ;;
  esac
done

if [[ ${#BUDGETS[@]} -eq 0 || ${#EPSILONS[@]} -eq 0 || ${#DELTAS[@]} -eq 0 ]]; then
  echo "INVALID_GRID: 无法从 $CONFIG_PATH 读取 sweep 网格" >&2
  exit 2
fi

total=0
success=0
failed=0

for epsilon in "${EPSILONS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      total=$((total + 1))
      eps_slug="$(slug_number "$epsilon")"
      delta_slug="$(slug_number "$delta")"
      budget_slug="$(slug_number "$budget")"
      output_csv="$POLICY_DIR/pilot_smoke_measured_policy_eps${eps_slug}_delta${delta_slug}_mem${budget_slug}.csv"
      cell_log="$LOG_DIR/policy_eps${eps_slug}_delta${delta_slug}_mem${budget_slug}.log"

      {
        echo "START $(date '+%F %T') epsilon=$epsilon delta=$delta memory_budget_mib=$budget"
        echo "OUTPUT $output_csv"
      } > "$cell_log"

      conda run -n "$CONDA_ENV" python -m run_util.run_policies \
        --config "$CONFIG_PATH" \
        --measurements "$MEASUREMENTS_PATH" \
        --output "$output_csv" \
        --epsilon "$epsilon" \
        --delta "$delta" \
        --memory-budget-mib "$budget" >> "$cell_log" 2>&1
      status=$?

      if [[ $status -eq 0 ]]; then
        echo "SUCCESS $(date '+%F %T') status=$status" >> "$cell_log"
        success=$((success + 1))
      else
        echo "FAILED $(date '+%F %T') status=$status" >> "$cell_log"
        failed=$((failed + 1))
      fi

      echo "CELL total=$total success=$success failed=$failed epsilon=$epsilon delta=$delta memory_budget_mib=$budget status=$status"
    done
  done
done

if [[ $failed -eq 0 ]]; then
  conda run -n "$CONDA_ENV" python "$ROOT_DIR/scripts/aggregate_baseline_wide_sweep.py" \
    --input-dir "$POLICY_DIR" \
    --output "$SUMMARY_PATH"
  aggregate_status=$?
else
  aggregate_status=1
fi

echo "SWEEP_DONE total=$total success=$success failed=$failed aggregate_status=$aggregate_status"

if [[ $failed -ne 0 || $aggregate_status -ne 0 ]]; then
  exit 1
fi
exit 0
