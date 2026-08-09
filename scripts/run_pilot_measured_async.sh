#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pilot.yaml}"
RUN_TAG="${2:-$(basename "$CONFIG_PATH" .yaml)}"
CONDA_ENV="${CONDA_ENV:-tailguardkv-base}"
LOG_DIR="$ROOT_DIR/out/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/${RUN_TAG}_${TS}.nohup.log"
PID_PATH="$LOG_DIR/${RUN_TAG}_${TS}.pid"
LATEST_PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

mkdir -p "$LOG_DIR"

nohup setsid conda run -n "$CONDA_ENV" python "$ROOT_DIR/run_experiment.py" pilot-smoke-measured --config "$CONFIG_PATH" \
  > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!

printf '%s\n' "$PID" > "$PID_PATH"
printf '%s\n' "$PID" > "$LATEST_PID_PATH"

echo "started pid=$PID"
echo "config=$CONFIG_PATH"
echo "log=$LOG_PATH"
echo "pidfile=$PID_PATH"
echo "latest_pidfile=$LATEST_PID_PATH"
