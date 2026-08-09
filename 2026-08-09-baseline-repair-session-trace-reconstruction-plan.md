# Baseline Repair + Session Trace Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 分两阶段修复 baseline 语义与会话级 measured replay 路径，让 baseline 表格可解释、backend 行为可验证。

**Architecture:** 第一阶段不改主请求夹具的数据形态，只修策略语义、fallback 统计和预算 sweep，让 baseline 的 action distribution、quality、memory 三组指标重新分化。第二阶段新增 session-aware trace 和单独配置，强制 measured replay 进入 resident、evict、restore、recompute 路径，验证会话累计与跨会话驻留逻辑。

**Tech Stack:** Python、JSONL fixtures、Markdown 文档、现有 policy/backend/metric collector 测试链路。

## Global Constraints

- 第一阶段继续使用 `pilot_qa_summary_requests.jsonl` 作为 baseline main fixture。
- 第一阶段不修改 `measured_replay` 对无 `session_id` 请求返回 baseline 的边界。
- `static_best` 保持 fixed-profile baseline 语义，不改成逐请求 adaptive。
- `utility_dynamic` 保持 `ttft + memory + pred_loss` 的综合 utility baseline 语义。
- 第二阶段优先新增独立 fixture 和独立 config，不替换第一阶段 main fixture。
- 第一阶段不要求出现真实的 `budget_hit / restore / recompute` 事件。
- 第二阶段必须把“策略层预算筛选”和“backend 全局预算撞线”区分开。

---

## File Structure

**预计修改/新增文件方向**

- 修改：策略实现文件
  - `static_safe`、`static_best`、`uncalibrated_dynamic`、`utility_dynamic` 所在 policy 模块
- 修改：指标汇总文件
  - `MetricCollector.summarize_policy_runs()` 所在模块
- 修改：baseline pilot/config 相关文件
  - 当前 measured pilot 配置、baseline sweep 配置、profile 标定配置引用处
- 新增：session-aware trace fixture
  - 放在现有 request fixtures 同层目录，保留 `qa/summary` 与 `reference`
- 新增：第二阶段独立配置
  - 指向 session-aware trace，不覆盖第一阶段配置
- 新增或修改：策略/后端/fixture loader 单测
  - 分别覆盖 Phase 1 和 Phase 2 的关键语义
- 修改：诊断或 summary 输出
  - 增补 session 统计、backend 事件、预算口径区分

---

### Task 1: 盘点现有实现与文件边界

**Files:**
- Modify: 无
- Create: 无
- Test: 无

**Interfaces:**
- Consumes: 仓库内现有 policy、backend、metrics、fixture loader、pilot config 实现
- Produces: 精确的修改清单，明确每个类/函数/配置入口所在文件

- [ ] **Step 1: 定位 policy、backend、metrics、fixture loader、pilot config 文件**

Run:

```bash
find . -type f | sort
```

Expected: 能定位 `static_safe`、`static_best`、`uncalibrated_dynamic`、`utility_dynamic`、`MetricCollector`、`MeasuredReplayBackend`、request loader、pilot config。

- [ ] **Step 2: 读取策略与汇总实现**

Run:

```bash
sed -n '1,240p' path/to/policy_module.py
sed -n '1,260p' path/to/metrics_module.py
```

Expected: 明确 `decide()`、`_candidate_action()`、`_best_exact_candidate()`、`fallback_reason` 汇总逻辑的现状。

- [ ] **Step 3: 读取 backend、fixture loader、pilot config**

Run:

```bash
sed -n '1,320p' path/to/measured_replay_backend.py
sed -n '1,220p' path/to/request_loader.py
sed -n '1,240p' path/to/pilot_config_or_runner.py
```

Expected: 明确第一阶段哪些边界保持不变，第二阶段新增 trace 和 config 应该挂在哪些入口。

- [ ] **Step 4: 记录修改映射**

输出内容至少包括：

```text
Policy file: ...
Metrics file: ...
Backend file: ...
Fixture loader file: ...
Pilot config file: ...
Existing tests: ...
```

Expected: 后续任务可以直接引用具体文件路径，不再模糊描述。

- [ ] **Step 5: Commit**

```bash
git status
```

Expected: 本任务只完成调研，不产生提交。

### Task 2: 为 `static_safe` 与 fallback 统计补失败测试

**Files:**
- Modify: `tests/...` 中对应 policy/metrics 测试文件
- Create: 如缺失则新增 `tests/.../test_policy_baselines.py`、`tests/.../test_metric_summary.py`
- Test: `tests/.../test_policy_baselines.py`、`tests/.../test_metric_summary.py`

**Interfaces:**
- Consumes: 现有 policy 实例化方式、候选 profile 构造方式、summary 输入记录结构
- Produces:
  - `static_safe` 在 `safe=False` 时必须回退 exact 的测试
  - 普通判断标签不计入 `fallback_ratio` 的测试

- [ ] **Step 1: 写 `static_safe` 的失败测试**

```python
def test_static_safe_falls_back_to_exact_when_fixed_profile_is_unsafe():
    policy = build_static_safe_policy(...)
    decision = policy.decide(request=unsafe_request, ...)
    assert decision.action.kind == "exact"
    assert decision.fallback_reason == "calibrated_unsafe"


def test_static_safe_keeps_lossy_when_fixed_profile_is_safe():
    policy = build_static_safe_policy(...)
    decision = policy.decide(request=safe_request, ...)
    assert decision.action.kind == "lossy"
    assert decision.fallback_reason is None


def test_static_best_keeps_fixed_lossy_profile_even_if_request_is_unsafe():
    policy = build_static_best_policy(...)
    decision = policy.decide(request=unsafe_request, ...)
    assert decision.action.kind == "lossy"
```

- [ ] **Step 2: 运行单测确认失败**

Run:

```bash
pytest tests/.../test_policy_baselines.py -v
```

Expected: 至少 `static_safe` fallback 相关断言失败。

- [ ] **Step 3: 写 fallback 统计失败测试**

```python
def test_summary_counts_only_real_fallback_reasons():
    runs = [
        make_run(policy="static_best", fallback_reason=None, safety_reason="calibrated_unsafe"),
        make_run(policy="static_safe", fallback_reason="calibrated_unsafe", safety_reason="calibrated_unsafe"),
    ]
    summary = MetricCollector().summarize_policy_runs(runs)
    assert summary["static_best"]["fallback_ratio"] == 0.0
    assert summary["static_safe"]["fallback_ratio"] == 1.0
```

- [ ] **Step 4: 运行汇总测试确认失败**

Run:

```bash
pytest tests/.../test_metric_summary.py -v
```

Expected: 当前实现会把 `safety_reason` 和 `fallback_reason` 混算，测试失败。

- [ ] **Step 5: Commit**

```bash
git add tests/.../test_policy_baselines.py tests/.../test_metric_summary.py
git commit -m "test: cover static safe fallback semantics"
```

### Task 3: 实现 `static_safe` 真回退与 fallback 语义拆分

**Files:**
- Modify: `path/to/policy_module.py`
- Modify: `path/to/metrics_record_or_schema.py`（如有记录对象）
- Modify: `path/to/metrics_module.py`
- Test: `tests/.../test_policy_baselines.py`、`tests/.../test_metric_summary.py`

**Interfaces:**
- Consumes:
  - `policy.decide(request, ...) -> decision`
  - `MetricCollector.summarize_policy_runs(runs) -> dict`
- Produces:
  - `decision.safety_reason: str | None`
  - `decision.fallback_reason: str | None`
  - `static_safe.decide()` 真实 exact fallback 语义

- [ ] **Step 1: 修改 `static_safe.decide()`**

```python
def decide(...):
    candidate = self._candidate_action(self._fixed_profile, request, ...)
    if candidate.safe:
        candidate.safety_reason = candidate.reason_annotation
        candidate.fallback_reason = None
        return candidate

    exact = self._best_exact_candidate(request, ...)
    exact.safety_reason = candidate.reason_annotation
    exact.fallback_reason = "calibrated_unsafe"
    return exact
```

Expected: `static_safe` 只在当前请求不安全时从固定 lossy profile 回退到 exact。

- [ ] **Step 2: 给普通判断标签单独字段**

```python
@dataclass
class PolicyDecision:
    ...
    safety_reason: str | None = None
    fallback_reason: str | None = None
```

Expected: “判断标签”和“真实动作替换原因”不再混用一个字段。

- [ ] **Step 3: 调整 summary 统计仅看 `fallback_reason`**

```python
fallback_count = sum(1 for run in runs if run.fallback_reason)
fallback_ratio = fallback_count / len(runs) if runs else 0.0
```

Expected: 原统计逻辑保留，但字段语义已经修正。

- [ ] **Step 4: 运行相关单测**

Run:

```bash
pytest tests/.../test_policy_baselines.py tests/.../test_metric_summary.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add path/to/policy_module.py path/to/metrics_module.py tests/.../test_policy_baselines.py tests/.../test_metric_summary.py
git commit -m "fix: repair static safe fallback semantics"
```

### Task 4: 为 `uncalibrated_dynamic` 与 `utility_dynamic` 补排序和预算测试

**Files:**
- Modify: `tests/.../test_policy_baselines.py` 或拆分成 `tests/.../test_dynamic_policies.py`
- Test: `tests/.../test_dynamic_policies.py`

**Interfaces:**
- Consumes: 动态 policy 的 profile 列表、预算过滤接口、score 公式
- Produces:
  - `uncalibrated_dynamic` 的稳定排序选择测试
  - `utility_dynamic` 的 score 与预算过滤测试

- [ ] **Step 1: 写 `uncalibrated_dynamic` 排序失败测试**

```python
def test_uncalibrated_dynamic_picks_best_sorted_eligible_lossy_profile():
    policy = build_uncalibrated_dynamic_policy(profiles=[p3, p1, p2], epsilon=0.05, budget=80)
    decision = policy.decide(request=req, ...)
    assert decision.profile_name == "p1"


def test_uncalibrated_dynamic_is_independent_of_profile_order():
    policy_a = build_uncalibrated_dynamic_policy(profiles=[p1, p2, p3], ...)
    policy_b = build_uncalibrated_dynamic_policy(profiles=[p3, p2, p1], ...)
    assert policy_a.decide(request=req, ...).profile_name == policy_b.decide(request=req, ...).profile_name
```

- [ ] **Step 2: 写 `utility_dynamic` 行为锁定测试**

```python
def test_utility_dynamic_picks_lowest_score_within_budget():
    policy = build_utility_dynamic_policy(profiles=[p1, p2, p3], budget=80, ...)
    decision = policy.decide(request=req, ...)
    assert decision.profile_name == "expected_lowest_score_profile"


def test_utility_dynamic_budget_filter_changes_candidate_set():
    loose = build_utility_dynamic_policy(profiles=[p1, p2, p3], budget=120, ...)
    tight = build_utility_dynamic_policy(profiles=[p1, p2, p3], budget=60, ...)
    assert loose.decide(request=req, ...).profile_name != tight.decide(request=req, ...).profile_name
```

- [ ] **Step 3: 运行测试确认当前实现失败或缺覆盖**

Run:

```bash
pytest tests/.../test_dynamic_policies.py -v
```

Expected: `uncalibrated_dynamic` 的顺序依赖测试失败，`utility_dynamic` 至少建立锁定覆盖。

- [ ] **Step 4: 检查 profile 构造是否可复用**

```python
def make_profile(name, pred_loss, ttft_ms, memory_mib, violation_rate=0.0):
    ...
```

Expected: 避免在多个测试里复制 profile fixture。

- [ ] **Step 5: Commit**

```bash
git add tests/.../test_dynamic_policies.py
git commit -m "test: cover dynamic baseline selection rules"
```

### Task 5: 实现 `uncalibrated_dynamic` 稳定排序并锁定 `utility_dynamic`

**Files:**
- Modify: `path/to/policy_module.py`
- Test: `tests/.../test_dynamic_policies.py`

**Interfaces:**
- Consumes:
  - `profiles`
  - `pred_loss <= epsilon`
  - 预算过滤逻辑
- Produces:
  - `uncalibrated_dynamic` 稳定排序选择逻辑
  - `utility_dynamic` 现有定义的回归保护

- [ ] **Step 1: 改 `uncalibrated_dynamic.decide()`**

```python
eligible = [p for p in profiles if fits_budget(p, request, budget)]
lossy = [p for p in eligible if p.kind != "exact" and p.pred_loss <= epsilon]
if not lossy:
    return self._best_exact_candidate(request, ...)

chosen = min(
    lossy,
    key=lambda p: (
        p.pred_loss,
        p.p95_ttft_ms,
        p.p95_kv_cache_memory_mib,
        p.name,
    ),
)
return self._candidate_action(chosen, request, ...)
```

- [ ] **Step 2: 确认 `utility_dynamic` 只在预算内候选打分**

```python
eligible = [p for p in profiles if fits_budget(p, request, budget)]
chosen = min(eligible, key=self._utility_score)
```

Expected: 不改 score 公式，只确保测试覆盖当前定义。

- [ ] **Step 3: 运行动态策略测试**

Run:

```bash
pytest tests/.../test_dynamic_policies.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: 跑关联测试防回归**

Run:

```bash
pytest tests/... -k "policy or dynamic or metric" -v
```

Expected: 与 policy/metrics 相关测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add path/to/policy_module.py tests/.../test_dynamic_policies.py
git commit -m "fix: stabilize dynamic baseline selection"
```

### Task 6: 重做第一阶段 baseline 预算 sweep 并完成端到端验收

**Files:**
- Modify: `path/to/pilot_config_or_runner.py`
- Modify: baseline sweep 配置文件或实验脚本
- Modify: 如有需要，补充结果说明文档
- Test: measured pilot 输出 summary/CSV

**Interfaces:**
- Consumes:
  - calibration profile 表中的 `p95_kv_cache_memory_mib`
  - 当前 baseline pilot 配置入口
- Produces:
  - 三档预算 sweep
  - 第一阶段端到端验收记录

- [ ] **Step 1: 从 calibration profile 表提取内存量级**

Run:

```bash
python - <<'PY'
import json, pathlib
path = pathlib.Path("path/to/calibration_profiles.json")
print(path.read_text()[:2000])
PY
```

Expected: 看清各 profile 的 `p95_kv_cache_memory_mib` 量级，避免继续使用 `4900/5000 MiB` 这类失真预算。

- [ ] **Step 2: 设计三档预算**

```text
宽松档: 允许多个 lossy 候选通过
中间档: 筛掉部分 lossy 候选
紧张档: 让部分请求触发 exact fallback，但不让所有策略完全退化
```

Expected: 预算与当前 profile 的实测量级同量级。

- [ ] **Step 3: 修改第一阶段 pilot 配置**

```yaml
memory_budget_mib:
  - <loose_budget>
  - <mid_budget>
  - <tight_budget>
```

Expected: 预算 sweep 只影响策略筛选，不试图在独立请求 trace 上制造 backend 撞预算。

- [ ] **Step 4: 运行 measured pilot**

Run:

```bash
python path/to/pilot_runner.py --config path/to/phase1_config.yaml
```

Expected: 至少一个 `epsilon / delta / budget` cell 中：
- `static_safe` 与 `static_best` 不同
- `uncalibrated_dynamic` 与 `utility_dynamic` 动作分布不同
- `memory_budget_mib` 变化会改变至少一条 policy 的 `action_distribution`

- [ ] **Step 5: 记录第一阶段验收结论**

记录内容至少包括：

```text
通过项:
- static_safe vs static_best 已分化
- fallback_ratio 不再虚高
- 动态 baseline 不再依赖 profile 顺序
- budget sweep 能影响 action distribution

不要求项:
- budget_hit_rate > 0
- restore_count > 0
- recompute_count > 0
```

- [ ] **Step 6: Commit**

```bash
git add path/to/pilot_config_or_runner.py path/to/phase1_config.yaml
git commit -m "chore: retune phase1 baseline budgets"
```

### Task 7: 为会话级 replay 与 fixture loader 补失败测试

**Files:**
- Modify: `tests/.../test_measured_replay_backend.py`
- Modify: `tests/.../test_request_loader.py`
- Create: 如缺失则新增上述测试文件
- Test: `tests/.../test_measured_replay_backend.py`、`tests/.../test_request_loader.py`

**Interfaces:**
- Consumes:
  - `MeasuredReplayBackend`
  - request fixture loader
- Produces:
  - session 连续 turn、跨 session 驱逐、restore/recompute 测试
  - session-aware JSONL loader 测试

- [ ] **Step 1: 写同 session 连续 turn 测试**

```python
def test_backend_accumulates_resident_kv_within_session():
    events = run_trace([
        req(session_id="s1", turn_index=0, arrival_index=0),
        req(session_id="s1", turn_index=1, arrival_index=1),
    ])
    assert events[1].resident_kv_mib_before == events[0].resident_kv_mib_after
```

- [ ] **Step 2: 写跨 session 预算驱逐与 restore/recompute 测试**

```python
def test_backend_evicts_or_offloads_when_global_budget_is_hit():
    events = run_trace(interleaved_requests(), budget_mib=...)
    assert any(e.budget_hit for e in events)
    assert any(e.event in {"evict_other_sessions", "offload_current_session"} for e in events)


def test_backend_adds_restore_or_recompute_cost_when_session_returns():
    events = run_trace(returning_session_requests(), budget_mib=...)
    assert any(e.restore_ms > 0 or e.recompute_ms > 0 for e in events)
```

- [ ] **Step 3: 写 loader 测试**

```python
def test_loader_preserves_session_fields_and_metadata():
    requests = load_requests("tests/fixtures/session_trace.jsonl")
    assert requests[0].session_id == "s1"
    assert requests[0].turn_index == 0
    assert requests[0].metadata["split"] == "pilot"
    assert requests[0].metadata["length_bucket"] == "medium"
```

- [ ] **Step 4: 运行测试确认失败**

Run:

```bash
pytest tests/.../test_measured_replay_backend.py tests/.../test_request_loader.py -v
```

Expected: 当前实现缺少 session-aware fixture 或未完全覆盖对应语义，测试失败。

- [ ] **Step 5: Commit**

```bash
git add tests/.../test_measured_replay_backend.py tests/.../test_request_loader.py
git commit -m "test: cover session replay behavior"
```

### Task 8: 新增 session-aware trace fixture 与第二阶段配置

**Files:**
- Create: `path/to/session_aware_requests.jsonl`
- Modify: `path/to/request_loader.py`（如需扩 schema）
- Create: `path/to/phase2_session_trace_config.yaml`
- Test: `tests/.../test_request_loader.py`

**Interfaces:**
- Consumes: 现有 request schema、loader、pilot config
- Produces:
  - session-aware JSONL fixture
  - 第二阶段独立 config

- [ ] **Step 1: 生成 session-aware trace fixture**

每条记录至少包含：

```json
{
  "request_id": "req-0001",
  "task": "qa",
  "prompt": "...",
  "reference": "...",
  "session_id": "session-a",
  "turn_index": 0,
  "metadata": {
    "source_dataset": "pilot",
    "split": "pilot",
    "length_bucket": "medium",
    "arrival_index": 0
  }
}
```

Expected: 同一 `session_id` 有连续 turn，不同 session 在 arrival order 上交错。

- [ ] **Step 2: 控制 trace 压力特征**

```text
至少 3-5 个 session
每个 session 至少 2-4 个 turn
长度与 profile 组合足以让全局 resident KV 靠近预算边界
task 仅保留 qa/summary
reference 保留质量标签能力
```

- [ ] **Step 3: 如有需要扩展 loader**

```python
Request(
    request_id=raw["request_id"],
    session_id=raw.get("session_id"),
    turn_index=raw.get("turn_index"),
    metadata=raw.get("metadata", {}),
)
```

Expected: `split`、`length_bucket`、`arrival_index` 不被覆盖或丢失。

- [ ] **Step 4: 新增 phase 2 config**

```yaml
requests_path: path/to/session_aware_requests.jsonl
memory_budget_mib:
  - <phase2_loose>
  - <phase2_mid>
  - <phase2_tight>
mode: measured_replay
```

Expected: phase 2 使用独立 config，不影响第一阶段主链路。

- [ ] **Step 5: 运行 loader 测试**

Run:

```bash
pytest tests/.../test_request_loader.py -v
```

Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add path/to/session_aware_requests.jsonl path/to/phase2_session_trace_config.yaml path/to/request_loader.py
git commit -m "feat: add session-aware replay fixture"
```

### Task 9: 实现第二阶段 backend 压力诊断与端到端验收

**Files:**
- Modify: `path/to/measured_replay_backend.py`
- Modify: `path/to/metrics_module.py` 或 summary/diagnostic 模块
- Test: `tests/.../test_measured_replay_backend.py`
- Test: phase 2 measured pilot 输出 summary/CSV

**Interfaces:**
- Consumes:
  - session-aware trace
  - backend resident / offload / drop / restore / recompute 逻辑
- Produces:
  - phase 2 backend 行为诊断
  - phase 2 summary 指标

- [ ] **Step 1: 对齐 backend 状态更新语义**

重点确认并修复：

```text
同 session 连续 turn 的 resident_kv_mib_before/after 衔接
profile 未切换时按 _projected_session_resident() 累积
offloaded session 返回时产生 restore
dropped session 返回时产生 recompute
多 session 交错时全局 resident KV 会竞争预算
```

- [ ] **Step 2: 补充 summary/diagnostic 字段**

至少输出：

```python
{
    "session_count": ...,
    "multi_turn_session_count": ...,
    "active_session_peak": ...,
    "budget_hit_rate": ...,
    "restore_count": ...,
    "recompute_count": ...,
    "mean_global_resident_kv_mib": ...,
    "p95_global_resident_kv_mib": ...,
}
```

Expected: 能明确区分 `policy_budget_filter_rate` 与 `budget_hit_rate`。

- [ ] **Step 3: 运行 backend 单测**

Run:

```bash
pytest tests/.../test_measured_replay_backend.py -v
```

Expected: PASS。

- [ ] **Step 4: 运行第二阶段 measured pilot**

Run:

```bash
python path/to/pilot_runner.py --config path/to/phase2_session_trace_config.yaml
```

Expected: 至少一个 budget cell 中：
- `budget_hit_rate > 0`
- 至少一个 policy 出现 `restore_count > 0` 或 `recompute_count > 0`
- `mean_global_resident_kv_mib` 与预算变化方向一致
- summary 与 raw policy CSV 可以互相解释

- [ ] **Step 5: 保留一个手工核对样本**

记录内容至少包括：

```text
reference cell: policy=..., budget=..., epsilon=..., delta=...
request sequence:
- req-...
- req-...
observed events:
- evict_other_sessions
- restore
- recompute
```

Expected: 便于对照 raw CSV 验证会话 trace 是否按设计触发事件。

- [ ] **Step 6: Commit**

```bash
git add path/to/measured_replay_backend.py path/to/metrics_module.py
git commit -m "feat: validate session replay pressure path"
```

## Self-Review

- **Spec coverage:** 本计划覆盖两阶段目标、关键接口、测试方案、端到端验收标准，以及第一阶段与第二阶段的边界区分。
- **Placeholder scan:** 仍需在执行 Task 1 后，把 `path/to/...` 和 `tests/...` 替换为真实文件路径；这是本计划唯一依赖仓库调研结果补全的部分。
- **Type consistency:** 计划内统一使用 `safety_reason` 与 `fallback_reason` 双字段，并统一使用 `session_id`、`turn_index`、`arrival_index` 作为 phase 2 trace 的关键字段。

## Execution Handoff

Plan complete and saved to `2026-08-09-baseline-repair-session-trace-reconstruction-plan.md`.

Two execution options:

1. Subagent-Driven (recommended) - 我按任务逐个分发和审查，实现过程更稳。
2. Inline Execution - 直接在当前会话里按这个计划继续落地修改。

## Completed Tasks

- 2026-08-09 已完成 `static_safe` 逐请求 exact fallback 语义修复：固定 profile 在当前请求 `safe=False` 时切到 exact；若初始化阶段已固定为 exact，则不再把它记成真实 fallback。
- 2026-08-09 已完成 fallback 口径拆分：新增 `safety_reason` 字段，`fallback_reason` 仅保留真实动作替换原因；`static_best` 与 `uncalibrated_dynamic` 的普通判断标签不再污染 fallback 统计。
- 2026-08-09 已完成 `uncalibrated_dynamic` 的稳定排序选择：不再依赖 profile 配置顺序，而是按 `pred_loss -> p95_ttft_ms -> projected_memory_mib -> profile name` 选择。
- 2026-08-09 已新增 phase 1 独立配置 `configs/pilot_phase1.yaml`，保留 `pilot_qa_summary_requests.jsonl` 主夹具，并将预算 sweep 缩到与当前 profile `p95_kv_cache_memory_mib` 同量级的 `[18, 26, 40]` MiB。
- 2026-08-09 已将 `configs/baseline_wide_sweep.yaml` 的预算 sweep 收敛到真实量级，当前网格为 `[18, 22, 26, 30, 35, 40, 50, 60, 75]` MiB。
- 2026-08-09 已新增 phase 2 session-aware trace 夹具 `data/fixtures/pilot_session_trace_requests.jsonl` 与独立配置 `configs/pilot_session_trace.yaml`，覆盖多 session 交错、multi-turn session、`qa/summary` 混合和显式 `arrival_index`。
- 2026-08-09 已扩展 summary 诊断字段：新增 `session_count`、`multi_turn_session_count`、`active_session_peak`、`triggered_restore`、`triggered_recompute`、`triggered_evict`、`triggered_queue`。
- 2026-08-09 已补齐并通过相关单测：覆盖 `static_safe` fallback、新的 fallback 统计语义、`uncalibrated_dynamic` 稳定排序、phase 1/phase 2 config、session trace loader、session summary 诊断。
