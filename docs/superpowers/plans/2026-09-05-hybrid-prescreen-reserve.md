# Hybrid Prescreen Reserve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit final-form hybrid selection when the LongBench `kivi_sensitive/qa` and `kivi_sensitive/summary` direct-prescreen reserves are 7 and 10, while all other direct-prescreen cells remain 12.

**Architecture:** Keep the CLI's scalar `--required-per-cell` as the base threshold. A small pure helper resolves the effective threshold for a `(risk_family, task)` cell; it returns 7 for `kivi_sensitive/qa` and 10 for `kivi_sensitive/summary` when the base is 12. Both the next-batch decision and terminal status use the helper so the supervisor and one-shot CLI cannot disagree.

**Tech Stack:** Python 3.10, pytest, existing JSON manifest contracts.

## Global Constraints

- Only direct-prescreen reserve eligibility changes; final hybrid classification still comes from final-form 8-profile measurements.
- `kivi_sensitive/qa` is 7 and `kivi_sensitive/summary` is 10 only when the configured base threshold is 12; all other cells use the configured base threshold.
- RAGhot remains QA-only; Summary cannot use RAGhot or tie fill.
- Final fixture still requires 48 five-turn sessions and four sessions in every `risk x task x split` cell.
- Existing `--required-per-cell 1` unit-test semantics remain uniform across every cell.

---

### Task 1: Add Cell-Specific Prescreen Threshold Resolution

**Files:**
- Modify: `../TailGuardKV-labeling/scripts/partition_labeling_batches.py`
- Modify: `../TailGuardKV-labeling/tests/test_labeling_batches.py`

**Interfaces:**
- Consumes: `counts: dict[tuple[str, str], int]` and the scalar `required_per_cell` accepted by the CLI.
- Produces: `_effective_required_per_cell(cell: tuple[str, str], base_required_per_cell: int) -> int`; `hybrid_need_status()` includes a `required_counts` mapping alongside current counts.

- [ ] **Step 1: Write failing threshold-resolution tests**

```python
def test_hybrid_need_status_relaxes_only_kivi_summary_at_the_default_threshold() -> None:
    completed = [_prescreen_with_counts({
        ("kivi_sensitive", "qa"): 12,
        ("kivi_sensitive", "summary"): 10,
        ("h2o_sensitive", "qa"): 12,
        ("h2o_sensitive", "summary"): 12,
        ("low_risk", "qa"): 12,
        ("low_risk", "summary"): 12,
    })]

    status = hybrid_need_status(manifests, completed, required_per_cell=12)

    assert status["status"] == "need_met"
    assert status["required_counts"]["kivi_sensitive/summary"] == 10


def test_hybrid_need_status_keeps_other_cells_at_twelve() -> None:
    completed = [_prescreen_with_counts({
        ("kivi_sensitive", "qa"): 11,
        ("kivi_sensitive", "summary"): 10,
        ("h2o_sensitive", "qa"): 12,
        ("h2o_sensitive", "summary"): 12,
        ("low_risk", "qa"): 12,
        ("low_risk", "summary"): 12,
    })]

    status = hybrid_need_status(manifests, completed, required_per_cell=12)

    assert status["status"] == "pending"
    assert status["required_counts"]["kivi_sensitive/qa"] == 12
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `conda run -n tailguardkv-base python -m pytest tests/test_labeling_batches.py -q`

Expected: FAIL because `hybrid_need_status()` still requires 12 for `kivi_sensitive/summary` and lacks `required_counts`.

- [ ] **Step 3: Implement the minimal resolver**

```python
RELAXED_PRESCREEN_REQUIRED_COUNTS = {
    ("kivi_sensitive", "summary"): 10,
}


def _effective_required_per_cell(
    cell: tuple[str, str], base_required_per_cell: int
) -> int:
    if base_required_per_cell == 12:
        return RELAXED_PRESCREEN_REQUIRED_COUNTS.get(cell, base_required_per_cell)
    return base_required_per_cell
```

Use this resolver in both `next_batch_for_hybrid_need()` and `hybrid_need_status()`. Return `required_counts` with string cell keys from `hybrid_need_status()`.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `conda run -n tailguardkv-base python -m pytest tests/test_labeling_batches.py -q`

Expected: PASS, including existing tests that pass `required_per_cell=1`.

- [ ] **Step 5: Commit**

```bash
git add ../TailGuardKV-labeling/scripts/partition_labeling_batches.py ../TailGuardKV-labeling/tests/test_labeling_batches.py
git commit -m "放宽混合预筛 Summary reserve"
```

### Task 2: Recompute the Exhausted Prescreen and Start Final Hybrid Selection

**Files:**
- Modify: `baseline_外部标注与混合全流程方案.md`
- Create: `../TailGuardKV-labeling/artifacts/external_baseline_20260904/prescreen/longbench/hybrid_need_status_relaxed.json`
- Create: `../TailGuardKV-labeling/artifacts/external_baseline_20260904/hybrid/hybrid_candidates_batch000.manifest.json`

**Interfaces:**
- Consumes: the 32 strict `.prescreen.json` manifests in `external_baseline_20260904`.
- Produces: an auditable relaxed status report and one `pending_measurement` 12-session hybrid batch, or an exact final-form insufficiency report.

- [ ] **Step 1: Recompute status without profile execution**

Run `partition_labeling_batches.py` against the existing 32 manifests with `--required-per-cell 12`; save its JSON output as `hybrid_need_status_relaxed.json`.

Expected: `status=need_met`, `required_counts.kivi_sensitive/summary=10`, and no GPU process starts.

- [ ] **Step 2: Select one final-form batch**

Run `select_hybrid_candidate_batches.py` against the new LongBench candidates, 32 manifests, completed prescreens, and unique ShareGPT skeletons. Do not pass `--source-exhausted` because the relaxed direct gate permits final-form selection.

Expected: `pending_measurement`, 12 sessions, 60 rows, 8 expected profiles, `profile_chunk_size=1`, and `use_persistent_workers=true`.

- [ ] **Step 3: Verify the manifest before GPU launch**

Verify each session has five turns, all skeleton/content/payload identities are unique, no RAGhot Summary row exists, and direct risk fields are absent from candidate payloads.

Expected: the manifest is safe to measure but contains no final risk labels.

- [ ] **Step 4: Launch the one hybrid batch asynchronously**

Use its existing persistent-worker dual-GPU configuration through `nohup setsid`; save PID and log under the new run root. Do not start a second GPU task.

- [ ] **Step 5: Update the plan ledger**

Record the approved one-cell relaxation, the exact `required_counts`, batch ID, source distributions, launch PID, and the unchanged prohibition on fixture import/smoke before final 48-session validation.
