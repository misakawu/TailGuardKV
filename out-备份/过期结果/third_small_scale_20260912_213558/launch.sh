#!/usr/bin/env bash
set +e
"/DATACENTER3/zhenxiang.wang/work/TailGuardKV/.worktrees/baseline-smoke-final-aggregation/out/third_small_scale_20260912_213558/run_baseline_wide_sweep.sh"
status=$?
printf '%s\n' "$status" > "/DATACENTER3/zhenxiang.wang/work/TailGuardKV/.worktrees/baseline-smoke-final-aggregation/out/third_small_scale_20260912_213558/coarse_sweep.exitcode"
exit "$status"
