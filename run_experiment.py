#!/usr/bin/env python3
# 启动示例:
# 1. 主质量 baseline_quality:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 2. 小规模质量 smoke:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml
# 3. session-aware baseline_session:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace.yaml
# 4. ShareGPT session/cache 诊断:
#    conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_sharegpt.yaml
from __future__ import annotations

# 常用运行命令:
# 1. profile runtime 预检:
#    python3 -m run_util.check_profiles --config configs/pilot_50.yaml --timeout 180
# 2. 快速质量轨道 measured gate，50 requests:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml
# 3. 主质量轨道 baseline_quality，200 requests:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 4. session-aware 语义轨道 baseline_session:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace.yaml
# 5. ShareGPT session/cache 诊断:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot_sharegpt.yaml
# 6. dry-run/CLI 兼容检查:
#    python3 -m run_util.build_profile_table --config configs/pilot.yaml --dry-run --output /tmp/tailguardkv_profiles.csv
#    python3 -m run_util.run_policies --config configs/pilot.yaml --measurements /tmp/tailguardkv_profiles.csv --output /tmp/tailguardkv_policy.csv --allow-dry-run-replay
# 7. 单个 policy 组合复跑:
#    python3 -m run_util.run_policies --config configs/pilot_50.yaml --measurements out/profile_tables/pilot_50_measured_profiles.csv --output /tmp/tailguardkv_policy_eps0p05_delta0p05_mem4900.csv --epsilon 0.05 --delta 0.05 --memory-budget-mib 4900
# 8. nohup + conda 后台跑主质量轨道:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml > out/logs/pilot_measured.nohup.log 2>&1 < /dev/null & echo $! > out/logs/pilot_measured.pid
# 9. nohup + conda 后台跑 session-aware 轨道:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot_session_trace.yaml > out/logs/pilot_session_trace.nohup.log 2>&1 < /dev/null & echo $! > out/logs/pilot_session_trace.pid
# 10. 查看后台实验状态和日志:
#    cat out/logs/pilot_measured.pid && ps -fp "$(cat out/logs/pilot_measured.pid)" && tail -n 120 out/logs/pilot_measured.nohup.log
# 11. 停止后台实验:
#    kill "$(cat out/logs/pilot_measured.pid)"

from run_util.experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
