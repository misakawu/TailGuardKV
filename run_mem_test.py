from __future__ import annotations

# run_mem_test 启动命令说明:
# 1. conda 前台运行显存预算扫描:
#    conda run -n tailguardkv-base python run_mem_test.py --base-config configs/pilot.yaml --max-requests 80 --budget-start-mib 10 --budget-stop-mib 100 --budget-step-mib 10
#
# 2. 当前实现入口后台运行显存预算扫描:
#    mkdir -p out/logs && python3 run_mem_test.py --base-config configs/pilot.yaml --max-requests 80 --budget-start-mib 10 --budget-stop-mib 100 --budget-step-mib 10 > out/logs/run_mem_test.log 2>&1 & echo $! > out/logs/run_mem_test.pid
#
# 说明:
# - 启动器只负责调用 run_util.mem_test.main()，实际显存预算扫描逻辑位于 run_util/mem_test.py。
# - 后台运行使用当前 shell 中的 python3；如需指定虚拟环境，请先激活环境或替换 python3 路径。
#
# 参数:
# - --base-config: 基础实验配置文件，默认 configs/pilot.yaml。
# - --run-dir: 本次运行输出目录；不传时使用 out/MM-DD-HH_mem_test。
# - --max-requests: 本次扫描最多使用的请求数，默认 80。
# - --budget-start-mib: 显存预算扫描起点，单位 MiB，默认 10。
# - --budget-stop-mib: 显存预算扫描终点，单位 MiB，默认 100。
# - --budget-step-mib: 显存预算扫描步长，单位 MiB，默认 10。
# - --include-tailguard: 将 tailguard 策略加入 baseline 扫描；默认不加入。
# - --total-summary-output: 指定 total summary CSV 输出路径；不传时自动从 smoke_summary 派生。

from run_util.mem_test import main


if __name__ == "__main__":
    raise SystemExit(main())
