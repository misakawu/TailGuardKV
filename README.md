# TailGuardKV

TailGuardKV：面向边缘大模型服务的尾部质量可控 KV 缓存自适应管理。

## 当前正式实验入口

当前 `run_experiment.py` 是正式实验的统一入口。现有实现只暴露一个子命令：

```bash
python3 run_experiment.py pilot-smoke-measured
```

该命令固定使用 `configs/pilot.yaml`，不接收 `--config`、`--output` 等顶层参数。执行流程为：

1. 调用 `run_build_profile_table.py` 的 `build_profile_table`，以真实 measured 模式生成 profile 表。
2. 读取并校验生成的 profile 表，要求配置中的全部 profile 都存在且 `measured=True`。
3. 调用 `run_run_policies.py` 的 `run_policies`，用同一 profile 表做 measured replay policy 评估。
4. 将完整 summary 写入宽表 CSV 文件。

固定输出位置为：

- `out/profile_tables/pilot_smoke_measured_profiles.csv`
- `out/policy_tables/pilot_smoke_measured_policy.csv`
- `out/policy_tables/pilot_smoke_measured_summary.csv`

summary 使用宽表 CSV，一行对应 `experiment`、单个 `profile` 或单个 `policy` 汇总对象。
左侧列优先放置 `section`、`name`、`ok`、`return_code`、`step`、`count`、`ok_count`、
`mean_ttft_ms`、`p95_ttft_ms`、`mean_peak_memory_mib`、`p95_peak_memory_mib`、
`mean_quality_loss`、`p95_quality_loss`、`violation_rate`、`delta_slack` 等关键对比指标。
配置加载或 profile 表校验错误返回 `2`；profile 或 policy 阶段运行失败时返回对应阶段的非零返回码。

底层 runner 仍可独立调用，适合调试或复现实验中间产物：

```bash
python3 run_build_profile_table.py --config configs/pilot.yaml --no-dry-run --output out/profile_tables/pilot_smoke_measured_profiles.csv
python3 run_run_policies.py --config configs/pilot.yaml --measurements out/profile_tables/pilot_smoke_measured_profiles.csv --output out/policy_tables/pilot_smoke_measured_policy.csv
```

`run_build_profile_table.py` 默认是 dry-run；正式 measured profile 需要显式传入
`--no-dry-run`。`run_run_policies.py` 默认拒绝 dry-run replay，只接受
`ok=True, measured=True` 且覆盖配置中全部 profile 的测量表；`--allow-dry-run-replay`
仅用于 smoke/debug。

## E0 三策略复现

E0 只验收 `full_gpu`、`kivi_4bit`、`h2o_heavy_hitter` 三类 profile，三者使用同一个
Qwen2.5 模型、同一组中等长度请求和同一条 quality_loss 计算链路：

```bash
conda run -n tailguardkv-base python run_check_profiles.py --config configs/e0_reproduce.yaml --output out/profile_tables/e0_profile_smoke.csv
conda run -n tailguardkv-base python run_build_profile_table.py --config configs/e0_reproduce.yaml --no-dry-run --output out/profile_tables/e0_measured_profiles.csv
conda run -n tailguardkv-base python run_run_policies.py --config configs/e0_reproduce.yaml --measurements out/profile_tables/e0_measured_profiles.csv --output out/policy_tables/e0_policy.csv
```

`run_run_policies.py` 默认要求 profile 表全部为 `ok=True, measured=True`，且每个请求都包含
配置中的三个 profile。`--allow-dry-run-replay` 仅保留给 smoke/debug。

## Pilot 资产准备

Pilot 默认使用 `/DATACENTER3/zhenxiang.wang/resource/Qwen2.5-7B-Instruct`，profile
smoke 使用 `/DATACENTER3/zhenxiang.wang/resource/TinyLlama-1.1B-Chat-v1.0`。请求数据由
LongBench QA/长上下文任务和 XSum 摘要任务组成：LongBench `qasper` 当前取到 200 条，
XSum validation 取到 300 条，均按原始顺序取有效样本：

```bash
conda run -n tailguardkv-base python env_asset_prepare/prepare_pilot_assets.py --download-models --download-data --hf-endpoint https://hf-mirror.com
conda run -n tailguardkv-base python env_asset_prepare/check_envs.py --json
conda run -n tailguardkv-base python run_check_profiles.py --config configs/pilot.yaml --timeout 180
conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured
```

如果当前网络可直接访问 Hugging Face，可省略 `--hf-endpoint`。

生成的请求文件为：

```text
/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/requests/longbench_xsum_pilot.jsonl
```
