# TTFT 冷启动修复与 40-Cell 交接文档

更新时间：2026-09-16

## 当前状态

已实现 persistent worker 的 `warm_profile` RPC，以及 `OnlineQwenSessionBackend.warm_profile()`。每个 policy 的首个已决 action 会在 shadow/真实执行前预热。预热只加载 runtime/model，不生成、不修改 online history、KV session reuse 或 `CacheState`。

worker PID、generation、启动耗时、模型加载耗时与预热状态仅作为 `extra_*` 审计列传递，不参与 `ttft_ms`、`latency_ms`、`recompute_ms` 或 KV 汇总。`PolicyRunRecord` 已保留这些 audit 字段，供 CSV 门禁使用。

三-session gate 已使用 `hybrid-session-002`、`003`、`004` 的全部 15 个原始请求，按原始 `arrival_index` 顺序对五策略执行，共 75 条记录。门禁现在只要求 TTFT/recompute 不超过对应 profile calibration p99 的 5 倍；用户已明确首条 TTFT 高于 3,000 ms 属于正常现象，因此不再使用 3,000 ms 硬阈值。

本轮已修复 H2O 在线多轮行为：H2O 不再复用 runtime past-KV，每轮从完整 online history 重建；返回的 H2O runtime cache 会释放，不保留在持久 worker。`static_safe` 支持 `fixed_profile`，registry 拒绝不存在或 exact profile；第四轮配置仅将它固定为 `kivi_4bit_residual64`，其他 policy 维持原候选逻辑。

## 代码与文档

- `profiles/persistent_worker.py`: 分发 `warm_profile` RPC。
- `profiles/qwen2_kv_runtime.py`: runtime/model 预热与 worker 审计元数据。
- `profiles/base.py`: Qwen2 worker 预热 payload helper。
- `backends/qwen_session.py`: backend 预热和审计元数据附加。
- `run_util/run_policies.py`: 每个 policy 首个已决 action 的预热调用。
- `run_util/core_types.py`: `PolicyRunRecord` 输出 `extra_worker_*`、`extra_warm_profile*` 审计列。
- `scripts/ttft_warm_gate.py`: 75-record fail-closed gate。
- `scripts/run_ttft_warm_replay.py`: 三-session GPU replay 与 gate 报告。
- `scripts/preflight_session25_fourth.py`: 两档预算审计；以 `global_budget_mib` 匹配 policy CSV，并将未出现压力事件标为 `expected_no_pressure_control`，而非阻断条件。
- `TTFT冷启动修复与40-Cell启动计划.md`: 已按 p99-only TTFT 条件更新。

## 已验证结果

- worker/backend 预热聚焦测试：`2 passed`。
- 相关 worker、backend、policy runtime、preflight 与 gate 测试：`50 passed`。
- audit schema 相关测试：`38 passed`。
- gate/replay 单元测试：`4 passed`。
- `git diff --check` 无输出。

本轮已通过的真实 GPU 三-session gate：

- records：`out/full_25session_baseline_9_14/full/policy_attempts/ttft_warm_gate_20260916_h2o_static_safe/ttft_warm_gate_records.csv`
- p99-only report：`out/full_25session_baseline_9_14/full/policy_attempts/ttft_warm_gate_20260916_h2o_static_safe/ttft_warm_gate_report.json`
- 结果：75 records，`passed: true`，无 worker restart/state loss、无启动字段混入服务列；`static_safe` 全部选择 `kivi_4bit_residual64`，无 H2O tensor/history 错误。

首条真实 TTFT 为约 3.7–4.1 s，但预热模型加载已被审计字段隔离；对应 calibration p99 的 5 倍限制内，用户确认这是正常现象。

## 已完成预算预检

两档预算 preflight 已完成，输出不混入 gate 或历史 attempt：

- log：`out/logs/tight_budget_preflight_20260916_h2o_static_safe.log`
- attempt root：`out/full_25session_baseline_9_14/full/policy_attempts/tight_budget_preflight_20260916_h2o_static_safe/`
- grid status：两个预算、10 个 policy cell 均为 `success`；共 650 条 records，全部 `ok=true`。
- 审计：`preflight_audit.json` 的 `passed=true`；最大 `global_resident_kv_mib` 为 81.7919921875 MiB，低于最小预算 120.2578125 MiB。`static_best`、`utility_dynamic` 和 `uncalibrated_dynamic` 在两档预算均为 `expected_no_pressure_control`，未出现真实执行错误或预算超限。

## 下一步

用户明确要求本会话不要启动 40-cell。后续会话如需启动，使用新的独立 attempt root，并在 GPU 进程运行期间不轮询：

```bash
nohup setsid bash -lc 'conda run --no-capture-output -n tailguardkv-base python scripts/run_session25_fourth_sweeps.py \
  --config configs/pilot_diagnostic_session25_fourth.yaml \
  --run-root out/full_25session_baseline_9_14/full \
  --attempt-root out/full_25session_baseline_9_14/full/policy_attempts/full40_20260916_h2o_static_safe' \
  > out/logs/full40_20260916_h2o_static_safe.log 2>&1 < /dev/null &
```

记录 PID 到 `out/logs/full40_20260916_h2o_static_safe.pid`。进程结束后，只聚合 `full40_20260916_h2o_static_safe`，生成线性/对数 TTFT 图、cell 汇总、事件汇总与最终审计报告。

## 保留规则

- 历史 `policy_attempts` 已从运行根可恢复地移动至 `/tmp/tailguardkv-policy-attempts-20260916`；不得将其中 CSV 重新混入新聚合。
- 不删除或覆盖 `merged`、原始 fixture、已有 out 目录和用户未提交改动。
- 正式聚合只能读取新 40-cell attempt root，不能混入历史失败/冷启动异常结果。
- 不得恢复 3,000 ms TTFT 硬阈值；当前批准门限是 per-profile calibration p99 的 5 倍。
