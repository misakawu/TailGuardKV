# 无压力 Cell 聚合与简化图表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接受无压力的有效 session policy 输出，并由完整 40-cell CSV 重建仅含策略曲线的可视化。

**Architecture:** 仅移除 baseline-session validator 的事件存在性条件，保留结构、成功状态、session 历史和预算验证。绘图函数停止叠加 session-level scatter，但保持入参兼容，使聚合调用不需改动。

**Tech Stack:** Python 3, pytest, CSV, Matplotlib。

## Global Constraints

- 不重跑 GPU 40-cell 实验。
- 只使用 `full40_20260916_h2o_static_safe` 中的 8 个 policy CSV。
- 最终 PNG 只显示策略曲线和置信区间。

---

### Task 1: 放宽无压力 Cell 的记录验证

**Files:**
- Modify: `run_util/experiment_semantics.py`
- Modify: `tests/test_experiment_semantics.py`

**Interfaces:**
- Consumes: `validate_experiment_policy_records(records, "baseline_session", output)`。
- Produces: 无压力但结构有效的 records 不抛出异常。

- [ ] **Step 1: Write the failing test**

```python
def test_session_policy_validation_accepts_no_pressure_control_records() -> None:
    validate_experiment_policy_records(
        [_record("r1", session_id="s1", turn_index=0, global_resident=10.0),
         _record("r2", session_id="s1", turn_index=1, global_resident=11.0)],
        "baseline_session",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiment_semantics.py::test_session_policy_validation_accepts_no_pressure_control_records -v`
Expected: failure citing missing backend pressure event.

- [ ] **Step 3: Write minimal implementation**

Remove only the backend-event-evidence rejection from the baseline session validation path.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `pytest tests/test_experiment_semantics.py -v`
Expected: all focused tests pass.

### Task 2: 移除图表 session 散点层

**Files:**
- Modify: `visual/plot_summary.py`
- Modify: `tests/test_tailguard_core.py`

**Interfaces:**
- Consumes: `plot_summary(summary_csv, session_points_csv=...)`。
- Produces: 曲线和置信区间，不产生 `PathCollection` scatter 图层。

- [ ] **Step 1: Write the failing test**

```python
assert not axis.collections
```

using a chart call with a valid session-points CSV.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tailguard_core.py -k session_points -v`
Expected: failure because scatter collections exist.

- [ ] **Step 3: Write minimal implementation**

Remove the `axis.scatter` branch while retaining the function signature.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `pytest tests/test_tailguard_core.py -k 'plot_summary or session_points' -v`
Expected: all selected tests pass.

### Task 3: 本地恢复与再聚合

**Files:**
- Modify: `scripts/run_session25_fourth_sweeps.py`
- Modify: `tests/test_session25_fourth_sweeps.py`

**Interfaces:**
- Consumes: a complete failed policy CSV without `split_seed`.
- Produces: a provenance-complete resumable CSV and an aggregate over all 8 cells.

- [ ] **Step 1: Write the failing test**

```python
assert _assert_attempt_output_available(path, 20260906, expected_ids)
assert "split_seed" in csv_header(path)
```

for a complete legacy CSV lacking only `split_seed`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session25_fourth_sweeps.py -k provenance_recovery -v`
Expected: failure because the existing helper raises incomplete output error.

- [ ] **Step 3: Write minimal implementation**

Recognize only the complete expected policy/request/diagnostic/online/ok CSV shape, append seed provenance, then revalidate it.

- [ ] **Step 4: Run focused tests and CPU recovery**

Run: `pytest tests/test_session25_fourth_sweeps.py -v` then rerun the sweep command against the existing attempt root.
Expected: all 8 cells success and regenerated summary/PNG outputs.
