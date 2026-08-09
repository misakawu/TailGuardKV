#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/out/logs"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

before_ps="$LOG_DIR/pilot_cleanup_${TS}_before.ps"
before_smi="$LOG_DIR/pilot_cleanup_${TS}_before.nvidia-smi"
after_ps="$LOG_DIR/pilot_cleanup_${TS}_after.ps"
after_smi="$LOG_DIR/pilot_cleanup_${TS}_after.nvidia-smi"

ps -eo pid,ppid,pgid,etime,cmd | grep -E 'run_experiment.py|conda run|profiles\.transformers_runtime|profiles\.qwen2_kv_runtime' | grep -v grep > "$before_ps" || true
nvidia-smi > "$before_smi" || true

pkill -f 'python run_experiment.py' || true
pkill -f 'conda run -n tailguardkv-base python run_experiment.py' || true
pkill -f 'profiles.transformers_runtime' || true
pkill -f 'profiles.qwen2_kv_runtime' || true

sleep 2

ps -eo pid,ppid,pgid,etime,cmd | grep -E 'run_experiment.py|conda run|profiles\.transformers_runtime|profiles\.qwen2_kv_runtime' | grep -v grep > "$after_ps" || true
nvidia-smi > "$after_smi" || true

echo "cleanup snapshots written:"
echo "  $before_ps"
echo "  $before_smi"
echo "  $after_ps"
echo "  $after_smi"
