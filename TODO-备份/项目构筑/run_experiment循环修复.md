# run_experiment Profile/Policy 闭环修复方案

## Summary

`run_experiment.py` 是薄编排层：`pilot_smoke_measured()` 依次做配置读取、profile 构建、profile 表校验、policy replay、summary 宽表写出。本轮修复同时补齐 profile chunk 失败诊断、`data.max_requests` split 分层取样、policy 参数 sweep 输出防覆盖和 summary 聚合。

## Key Changes

- `data_utils.py`
  - 增加 `limit_requests_by_split(requests, max_requests)`。
  - calibration/eval 都存在且 `max_requests >= 2` 时按 1:1 分层取样。
  - 单 split、缺 split、`max_requests <= 0` 保持原有语义。
  - 在 `experiment_common.py` 同步导出。

- `run_build_profile_table.py`
  - 用 `limit_requests_by_split()` 替换前缀截断。
  - 增加 `_failed_chunks_output()`，输出 `<stem>_failed_chunks.csv`。
  - chunk 校验失败时，主 profile CSV 只保留成功 chunk，失败 chunk 写入 sidecar CSV。
  - stdout JSON 返回 `ok=False`、`output`、`diagnostic_output`、`error`、`failures`。
  - 不改变 dry-run、import-measurements、chunk progress stderr 行为。

- `run_experiment.py`
  - `_run_stage()` 继续只捕获 stdout JSON，不捕获 stderr progress。
  - 增加 `_number_list()`、`_policy_sweep_points()`、`_policy_output_for_sweep()`。
  - profile 阶段仍只运行一次。
  - profile 失败或二次校验失败时不调用 policy，并把 `diagnostic_output`、`failures` 写入 summary。
  - policy 阶段展开 `pilot.epsilons x pilot.deltas x pilot.memory_budgets_mib`。
  - 单组合继续写 `outputs.smoke_policy`；多组合写参数后缀 CSV。
  - `rows.policy` 为已执行组合总行数。
  - payload 增加 `policy_runs`，每个元素包含 `ok`、`return_code`、`epsilon`、`delta`、`memory_budget_mib`、`output`、`payload`。
  - `_summary_rows()` 在 `policy_runs` 存在时按 sweep 展开 policy summary。
  - `SUMMARY_KEY_COLUMNS` 增加 `diagnostic_output`、`failures`。

- `run_run_policies.py`
  - 不修改 CLI 和内部语义。
  - 继续只跑一个 epsilon/delta/memory budget 组合。

- `README.md`
  - 说明 split 分层取样、失败 chunk 诊断文件、多组合 policy CSV 命名和 summary 聚合。
  - 记录 `full_cpu` 内存口径后续单独确认。

## Tasks

- [x] Task 1: 分层限制 `max_requests`
  - [x] 增加 `limit_requests_by_split()` 测试。
  - [x] 实现 helper 并从 `experiment_common.py` 导出。
  - [x] 更新 profile builder 的 mixed split max_requests 测试。
  - [x] 运行目标测试。

- [x] Task 2: 失败 chunk 诊断落盘
  - [x] 扩展 `test_build_profile_table_chunk_failure_keeps_completed_output`。
  - [x] 实现 `_failed_chunks_output()`。
  - [x] chunk 校验失败时写 sidecar 并返回诊断 JSON。
  - [x] 运行目标测试。

- [x] Task 3: `run_experiment` 透传 profile 诊断
  - [x] 扩展 profile stage 失败测试。
  - [x] summary 关键列增加 `diagnostic_output`、`failures`。
  - [x] profile 失败 payload 提升诊断字段。
  - [x] 运行目标测试。

- [x] Task 4: policy sweep 输出不覆盖
  - [x] 增加 helper 测试。
  - [x] 实现 sweep 组合和输出路径 helper。
  - [x] 增加 `pilot_smoke_measured` sweep 测试。
  - [x] 将单次 policy 调用替换为 sweep 循环。
  - [x] 运行目标测试。

- [x] Task 5: summary 聚合每个 policy sweep
  - [x] 增加 summary 展开测试。
  - [x] `_summary_rows()` 按 `policy_runs` 展开每个组合。
  - [x] 保持单组合路径兼容。
  - [x] 运行目标测试。

- [x] Task 6: 文档和回归验证
  - [x] 替换本修复计划文档。
  - [x] 更新 README 实验入口说明。
  - [x] 运行完整测试。
  - [x] 运行 dry-run/CLI 兼容检查。

## Verification Commands

```bash
python3 -m pytest tests/test_tailguard_core.py -v
python3 run_build_profile_table.py --config configs/pilot.yaml --dry-run --output /tmp/tailguardkv_profiles.csv
python3 run_run_policies.py --config configs/pilot.yaml --measurements /tmp/tailguardkv_profiles.csv --output /tmp/tailguardkv_policy.csv --allow-dry-run-replay
```

Measured 验收命令仍是：

```bash
python3 run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml
```

Measured 验收标准：

- 若 KIVI/H2O 仍失败，summary 必须能定位 `diagnostic_output` 和失败 error。
- 若 profile 全部成功，profile CSV 应包含 10 个 profile x 50 request。
- `split` 应同时包含 calibration 和 eval。
- 多组合 policy CSV 不互相覆盖，summary 聚合所有组合。

## Assumptions

- 不在本轮修改 KIVI/H2O runtime 逻辑；本轮先让失败原因可见、可复跑。
- 不修改 `run_run_policies.py` 的 public CLI 行为。
- 多组合中某个 policy run 失败时，停止后续组合，summary 保留已执行组合和失败组合 payload，返回失败组合 return code。
- 不把失败 chunk 混入主 profile CSV；主表继续只作为 measured replay 的有效输入。
