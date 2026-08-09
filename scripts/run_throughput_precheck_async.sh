#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT_DIR/scripts/run_pilot_measured_async.sh" "$ROOT_DIR/configs/pilot_throughput_precheck.yaml" "pilot_throughput_precheck"
