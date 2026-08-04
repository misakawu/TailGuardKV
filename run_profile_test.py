from __future__ import annotations

# 常用运行命令:
# 1. conda 前台运行 profile test:
#    conda run -n tailguardkv-base python run_profile_test.py --config configs/pilot_50.yaml > out/logs/profile_test.nohup.log 2>&1 < /dev/null & echo $! > out/logs/profile_test.pid
# 2. nohup + conda 后台运行 profile test:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_profile_test.py --config configs/pilot_50.yaml > out/logs/profile_test.nohup.log 2>&1 < /dev/null & echo $! > out/logs/profile_test.pid

from run_util.profile_test import main


if __name__ == "__main__":
    raise SystemExit(main())
