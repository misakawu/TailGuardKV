# TailGuardKV-
TailGuardKV：面向边缘大模型服务的尾部质量可控 KV 缓存自适应管理

## 7月代码骨架

统一入口是 `run_experiment.py`：

```bash
python3 run_experiment.py check-profiles --timeout 120
python3 run_experiment.py build-profile-table --output out/profile_tables/smoke_profiles.csv --dry-run
python3 run_experiment.py run-policies --measurements out/profile_tables/smoke_profiles.csv --allow-dry-run-replay
python3 run_experiment.py reproduce-profiles --config configs/pilot.yaml --no-dry-run
```

`check-profiles` 复现 full/KIVI/H2O adapter 的环境与关键模块导入；`build-profile-table`
生成统一 `request x profile` 表；`run-policies` 强制所有策略经过同一
`MeasuredReplayBackend` 和同一指标代码。默认 replay 只接受 `measured=True` 的实测
profile 表；`--allow-dry-run-replay` 仅用于 smoke。

`reproduce-profiles` 默认尝试真实 transformers profile；`build-profile-table --dry-run`
只验证表结构，质量指标会标为 unknown。已有外部实测 CSV 可通过
`build-profile-table --import-measurements path/to/measured.csv` 导入。

## E0 三策略复现

E0 只验收 `full_gpu`、`kivi_4bit`、`h2o_heavy_hitter` 三类 profile，三者使用同一个
Qwen2.5 模型、同一组中等长度请求和同一条 quality_loss 计算链路：

```bash
conda run -n tailguardkv-base python run_experiment.py check-profiles --config configs/e0_reproduce.yaml --output out/profile_tables/e0_profile_smoke.csv
conda run -n tailguardkv-base python run_experiment.py build-profile-table --config configs/e0_reproduce.yaml --no-dry-run --output out/profile_tables/e0_measured_profiles.csv
conda run -n tailguardkv-base python run_experiment.py run-policies --config configs/e0_reproduce.yaml --measurements out/profile_tables/e0_measured_profiles.csv --output out/policy_tables/e0_policy.csv
```

`run-policies` 默认要求 profile 表全部为 `ok=True, measured=True`，且每个请求都包含
配置中的三个 profile。`--allow-dry-run-replay` 仅保留给 smoke/debug。

## Pilot 资产准备

Pilot 默认使用 `/DATACENTER3/zhenxiang.wang/resource/Qwen2.5-7B-Instruct`，profile
smoke 使用 `/DATACENTER3/zhenxiang.wang/resource/TinyLlama-1.1B-Chat-v1.0`。请求数据由
LongBench QA/长上下文任务和 XSum 摘要任务组成：LongBench `qasper` 当前取到 200 条，
XSum validation 取到 300 条，均按原始顺序取有效样本：

```bash
conda run -n tailguardkv-base python env_asset_prepare/prepare_pilot_assets.py --download-models --download-data --hf-endpoint https://hf-mirror.com
conda run -n tailguardkv-base python env_asset_prepare/check_envs.py --json
conda run -n tailguardkv-base python run_experiment.py check-profiles --config configs/pilot.yaml --timeout 180
conda run -n tailguardkv-base python run_experiment.py build-profile-table --config configs/pilot.yaml --dry-run --output out/profile_tables/pilot_dry_profiles.csv
conda run -n tailguardkv-base python run_experiment.py run-policies --config configs/pilot.yaml --measurements out/profile_tables/pilot_dry_profiles.csv --allow-dry-run-replay
```

如果当前网络可直接访问 Hugging Face，可省略 `--hf-endpoint`。

生成的请求文件为：

```text
/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/requests/longbench_xsum_pilot.jsonl
```
