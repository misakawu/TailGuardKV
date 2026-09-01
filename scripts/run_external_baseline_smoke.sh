#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK="${1:-quality}"
CONDA_ENV="${CONDA_ENV:-tailguardkv-base}"

case "$TRACK" in
  quality)
    CONFIG_PATH="$ROOT_DIR/configs/pilot_external_baseline_quality.yaml"
    RUN_TAG="pilot_external_baseline_quality"
    ;;
  session)
    CONFIG_PATH="$ROOT_DIR/configs/pilot_external_baseline_session.yaml"
    RUN_TAG="pilot_external_baseline_session"
    ;;
  *)
    echo "unknown track: $TRACK" >&2
    echo "usage: $0 [quality|session]" >&2
    exit 2
    ;;
esac

"$ROOT_DIR/scripts/run_pilot_measured_async.sh" "$CONFIG_PATH" "$RUN_TAG"
