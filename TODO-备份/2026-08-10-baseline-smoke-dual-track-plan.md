# Baseline Smoke 双轨修复与硬门禁实施计划

## 目标

在 2026-08-10 当前代码基础上，将 baseline smoke 明确拆成两条正式轨道：

- `baseline_quality`：使用 `data/fixtures/pilot_qa_summary_requests.jsonl`，只用于 5 条 baseline 的质量、TTFT 和 KV 分化，不宣称 backend/session 事件语义。
- `baseline_session`：使用 `data/fixtures/pilot_session_trace_requests.jsonl`，用于验证 session 复用、resident/global resident、预算命中、restore/recompute/evict/queue 等 backend 语义。

session 轨道必须执行按实验类型的硬门禁；缺少 session 或 backend 证据时，流程返回失败，而不是以“流程完成”代替语义完成。

## 实施范围

1. 配置层增加显式实验类型，并保留旧 `data.quality_mode` 的兼容映射。
2. 请求、profile、policy/backend records 和 summary 各层增加实验类型校验。
3. 质量轨道的 backend 字段显式标记为不适用，避免空值被解释为真实零事件。
4. session 轨道检查多 session、多 turn、resident 字段、global resident 演化和至少一种真实 backend 压力事件。
5. 保持质量 fixture 与 session fixture 分离，不把独立质量请求自动拼接成 session。
6. 更新配置、summary/diagnostic 输出、状态文档和 baseline 说明。
7. 补充回归测试，并运行 pytest 与可执行的 smoke/配置级验证。

## 验收标准

- 主质量 baseline 表不再被解释成 session/backend smoke。
- session-aware smoke 对缺少 session/backend 语义的结果硬失败。
- session-aware smoke 至少展示一组预算和 backend 事件联动。
- summary、diagnostic、配置和状态文档使用一致的 `baseline_quality` / `baseline_session` 口径。
- 根目录文档末尾追加本轮完成任务和 `pytest.ini` 解决任务分析。

## 实施记录

### 已完成任务

- 已在 [`configs/pilot.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot.yaml)、[`configs/pilot_50.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_50.yaml)、[`configs/pilot_phase1.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_phase1.yaml)、[`configs/pilot_session_trace.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_session_trace.yaml) 和 [`configs/pilot_sharegpt.yaml`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/configs/pilot_sharegpt.yaml) 增加显式 `experiment.type`，把质量轨道固定为 `baseline_quality`，把 session 轨道固定为 `baseline_session`。
- 已在 [`run_util/config_loader.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/config_loader.py) 增加 `config_experiment_type()`，并保留对旧 `data.quality_mode` 的兼容映射；没有显式 session 声明的历史测试配置默认回落到质量轨道，避免误触发硬门禁。
- 已在 [`run_util/data_utils.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/data_utils.py) 增加 `validate_requests_for_experiment_type()`：`baseline_session` 现在强制要求非空 `session_id`、至少一个 `turn_index > 0`、至少两个 session，以及至少一个多 turn session。
- 已在 [`run_util/validation.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/validation.py) 增加 experiment-type-aware 校验：session 轨道强制要求 profile 表包含 `resident_kv_mib_before/after`、`kv_cumulative_mib` 和 `session_id`；policy records 缺少 `global_resident_kv_mib` 演化或缺少 `budget_hit/evict/restore/recompute/queue` 证据时直接失败。
- 已在 [`run_util/build_profile_table.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/build_profile_table.py)、[`run_util/run_policies.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/run_policies.py) 和 [`run_util/experiment.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/experiment.py) 接入实验类型，确保返回码反映语义失败，而不是只反映流程是否崩溃。
- 已在 [`metrics/collector.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/metrics/collector.py) 和 [`run_util/experiment_summary.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/run_util/experiment_summary.py) 补双轨 summary 语义：
  质量轨道把 `budget_hit_rate`、`restore_count`、`recompute_count`、`queue_delay_ms`、`mean_global_resident_kv_mib` 等字段标记为不适用；
  session 轨道输出 `backend_events_applicable`、`backend_semantics_status`、`session_reuse_evidence`、`global_resident_evolution`、`backend_event_evidence` 等检查结果。
- 已新增 [`tests/test_experiment_semantics.py`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/tests/test_experiment_semantics.py)，覆盖实验类型识别、质量/会话请求门禁、session profile 校验、质量 summary `N/A` 语义以及 session backend 事件硬门禁。
- 已更新 [`README.md`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/README.md)、[`README-当前实现对照.md`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/README-当前实现对照.md)、[`理想baseline_smoke表说明.md`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/理想baseline_smoke表说明.md) 和 [`docs/status/2026-08-15_baseline_smoke_status.md`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/docs/status/2026-08-15_baseline_smoke_status.md)，统一为双轨口径。

### 验证结果

- `pytest -q`：`284 passed, 3 subtests passed`。
- `pytest -q tests/test_experiment_semantics.py tests/test_pilot_dataset.py tests/test_session_refactor.py tests/test_session_summary_visuals.py`：`29 passed`。
- `python -m py_compile run_util/config_loader.py run_util/data_utils.py run_util/validation.py run_util/build_profile_table.py run_util/run_policies.py run_util/experiment.py metrics/collector.py run_util/experiment_summary.py`：通过。
- 配置检查结果：
  - `configs/pilot.yaml -> baseline_quality`
  - `configs/pilot_session_trace.yaml -> baseline_session`
  - `configs/pilot_sharegpt.yaml -> baseline_session`

### `pytest.ini` 解决了哪些任务

当前 [`pytest.ini`](/DATACENTER3/zhenxiang.wang/work/TailGuardKV/pytest.ini) 内容是：

- `testpaths = tests`
- `pythonpath = .`
- `norecursedirs = third_party .git __pycache__`

它在本轮实际解决了这些任务：

- 保证 `pytest` 默认只收集 `tests/`，避免把 `third_party/` 里的上游样例、临时脚本和外部仓库测试误收进本轮回归。双轨修复涉及全量 `pytest -q`，如果没有这条，外部目录会放大噪声，无法把失败定位到本仓实现。
- 保证仓库根目录在 `PYTHONPATH` 中，使 `run_util`、`metrics`、`backends`、`policies` 等顶层模块能被测试直接导入。本轮新增的实验类型测试和既有 `test_tailguard_core.py` 都依赖这一点。
- 通过 `norecursedirs` 排除 `.git` 和 `__pycache__`，减少无关扫描；这对多次全量回归时的稳定性和速度有直接帮助。

它没有直接解决的任务也需要明确：

- `pytest.ini` 不会替代实验语义硬门禁；`baseline_quality` / `baseline_session` 的区别仍然由代码中的 experiment-type 校验负责。
- `pytest.ini` 不会决定 session 轨道是否真的产生 backend 事件；这仍然依赖 fixture、预算和 replay 逻辑本身。
