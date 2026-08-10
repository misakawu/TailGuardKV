#!/usr/bin/env python3
# run_experiment.py 启动说明
# 1. 先做 profile runtime 预检，避免主实验中途因为环境/profile 不可用而失败:
#    conda run -n tailguardkv-base python -m run_util.check_profiles --config configs/pilot_50.yaml --timeout 180
# 2. 直接启动主入口，子命令固定为 pilot-smoke-measured，配置决定跑哪条实验轨道:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config <CONFIG>
# 3. 常用配置:
#    configs/pilot_50.yaml                小规模质量 smoke，快速检查 measured 链路
#    configs/pilot.yaml                   主质量轨道 baseline_quality
#    configs/pilot_session_trace.yaml     session-aware 语义轨道 baseline_session
#    configs/pilot_sharegpt.yaml          ShareGPT session/cache 诊断
# 4. 直接前台启动示例:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace.yaml
# 5. 后台启动示例:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml > out/logs/pilot_measured.nohup.log 2>&1 < /dev/null & echo $! > out/logs/pilot_measured.pid
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace.yaml > out/logs/pilot_session_trace.nohup.log 2>&1 < /dev/null & echo $! > out/logs/pilot_session_trace.pid
# 6. 查看后台状态:
#    cat out/logs/pilot_measured.pid
#    ps -fp "$(cat out/logs/pilot_measured.pid)"
#    tail -n 120 out/logs/pilot_measured.nohup.log
# 7. 停止后台任务:
#    kill "$(cat out/logs/pilot_measured.pid)"
# 8. 仅复查单独阶段时，可直接调用底层 CLI:
#    conda run -n tailguardkv-base python -m run_util.build_profile_table --config configs/pilot.yaml --dry-run --output /tmp/tailguardkv_profiles.csv
#    conda run -n tailguardkv-base python -m run_util.run_policies --config configs/pilot.yaml --measurements /tmp/tailguardkv_profiles.csv --output /tmp/tailguardkv_policy.csv --allow-dry-run-replay

from __future__ import annotations

from run_util.experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
