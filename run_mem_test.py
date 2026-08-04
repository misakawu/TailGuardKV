from __future__ import annotations

# run_mem_test 启动命令说明:
# 1. conda 前台运行显存预算扫描:
#    conda run -n tailguardkv-base python run_mem_test.py --base-config configs/pilot.yaml --max-requests 80 --budget-start-mib 100 --budget-stop-mib 5000 --budget-step-mib 100
#
# 2. nohup + conda 虚拟环境后台运行显存预算扫描:
#    mkdir -p out/logs && setsid nohup /DATACENTER3/zhenxiang.wang/miniforge3/envs/tailguardkv-base/bin/python run_mem_test.py --base-config configs/pilot.yaml --max-requests 80 --budget-start-mib 100 --budget-stop-mib 5000 --budget-step-mib 100 > out/logs/run_mem_test.nohup.log 2>&1 < /dev/null & echo $! > out/logs/run_mem_test.pid
#
# 说明:
# - 后台长任务使用 setsid + nohup 脱离当前会话，避免执行器退出时清理后台进程。
# - 使用 tailguardkv-base 环境中的 python 解释器，等价于在该 conda 虚拟环境中运行主 runner。
#
# 参数:
# - --base-config: 基础实验配置文件，默认 configs/pilot.yaml。
# - --run-dir: 本次运行输出目录；不传时使用 out/YYYYMMDD_mem_test。
# - --max-requests: 本次扫描最多使用的请求数，默认 80。
# - --budget-start-mib: 显存预算扫描起点，单位 MiB，默认 100。
# - --budget-stop-mib: 显存预算扫描终点，单位 MiB，默认 5000。
# - --budget-step-mib: 显存预算扫描步长，单位 MiB，默认 100。
# - --include-tailguard: 将 tailguard 策略加入 baseline 扫描；默认不加入。
# - --total-summary-output: 指定 total summary CSV 输出路径；不传时自动从 smoke_summary 派生。

from run_util.mem_test import main


if __name__ == "__main__":
    raise SystemExit(main())
