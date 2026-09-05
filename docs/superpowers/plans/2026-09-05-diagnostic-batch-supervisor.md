# Diagnostic Batch Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute all isolated diagnostic batches, record every outcome, and emit merged diagnostic-only artifacts only after complete validation.

**Architecture:** Extend the existing batch runner with pure validation and merge helpers. The supervisor invokes each child serially, writes per-batch status into a top-level manifest, and continues after nonzero child exits. Gate-only exits can be accepted only with complete, successful profile coverage; other failures prevent creation of merged artifacts.

**Tech Stack:** Python 3.10, standard-library `csv`/`json`/`subprocess`, PyYAML, pytest.

## Global Constraints

- GPU work remains serial and is launched only by the existing async/nohup wrapper.
- Every generated artifact and manifest has `diagnostic_only: true`.
- A merge must reject missing, failed, duplicate, or incomplete `(request_id, profile)` coverage.
- Gate-only failures do not short-circuit later batches; execution failures remain non-mergeable.
- No partial merge output is written when any batch is non-mergeable.

---

### Task 1: Batch Artifact Validation

**Files:**
- Modify: `scripts/run_diagnostic_session_batches.py`
- Test: `tests/test_diagnostic_batch_runner.py`

**Interfaces:**
- Produces: `validate_batch_output(batch: SessionBatch, run_dir: Path, profile_names: set[str]) -> dict[str, Any]`.
- Produces: `is_diagnostic_gate_failure(run_dir: Path) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_batch_output_accepts_complete_measured_profile_coverage(tmp_path: Path) -> None:
    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})
    assert status["mergeable"] is True
    assert status["profile_rows"] == 4

def test_validate_batch_output_rejects_duplicate_profile_coverage(tmp_path: Path) -> None:
    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})
    assert status["mergeable"] is False
    assert "duplicate" in status["errors"][0]
```

- [ ] **Step 2: Run the targeted tests and confirm the missing helper failure**

Run: `conda run --no-capture-output -n tailguardkv-base pytest -q tests/test_diagnostic_batch_runner.py`

- [ ] **Step 3: Implement minimal CSV and gate-artifact validation**

```python
def validate_batch_output(batch: SessionBatch, run_dir: Path, profile_names: set[str]) -> dict[str, Any]:
    # Verify one successful measured row for each request/profile key.
    ...
```

- [ ] **Step 4: Re-run the targeted tests and confirm they pass**

Run: `conda run --no-capture-output -n tailguardkv-base pytest -q tests/test_diagnostic_batch_runner.py`

### Task 2: Serial Supervisor Status and Merge

**Files:**
- Modify: `scripts/run_diagnostic_session_batches.py`
- Test: `tests/test_diagnostic_batch_runner.py`

**Interfaces:**
- Consumes: `validate_batch_output` and `is_diagnostic_gate_failure` from Task 1.
- Produces: `run_batches(manifest: dict[str, Any], root: Path, repo_root: Path, conda_env: str, runner: Callable[..., int]) -> int`.
- Produces: `merge_batch_outputs(statuses: list[dict[str, Any]], root: Path) -> Path`.

- [ ] **Step 1: Write failing supervisor tests**

```python
def test_run_batches_continues_after_complete_gate_failure_and_merges(tmp_path: Path) -> None:
    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)
    assert calls == ["batch000", "batch001"]
    assert exit_code == 0
    assert (root / "merged" / "profile_tables" / "diagnostic_session27_profiles.csv").exists()

def test_run_batches_writes_status_but_refuses_partial_merge(tmp_path: Path) -> None:
    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)
    assert exit_code == 1
    assert not (root / "merged").exists()
    assert json.loads((root / "supervisor_manifest.json").read_text())["merged"] is False
```

- [ ] **Step 2: Run the targeted tests and confirm they fail for the missing supervisor helper**

Run: `conda run --no-capture-output -n tailguardkv-base pytest -q tests/test_diagnostic_batch_runner.py`

- [ ] **Step 3: Implement serial execution, status persistence, and strict merge**

```python
for item in manifest["batches"]:
    returncode = runner(item)
    status = validate_batch_output(...)
    status["returncode"] = returncode
    status["gate_only_failure"] = returncode != 0 and is_diagnostic_gate_failure(...)
    statuses.append(status)
write_supervisor_manifest(statuses)
if all(status["mergeable"] for status in statuses):
    merge_batch_outputs(statuses, root)
    return 0
return 1
```

- [ ] **Step 4: Re-run the targeted tests and confirm they pass**

Run: `conda run --no-capture-output -n tailguardkv-base pytest -q tests/test_diagnostic_batch_runner.py`

### Task 3: Integration Verification and Async Run

**Files:**
- Modify: `smoke表降级进度.md`

**Interfaces:**
- Consumes: the supervisor CLI from Task 2.
- Produces: a recorded async diagnostic run root and final supervisor manifest.

- [ ] **Step 1: Add the updated supervisor/merge contract to the handoff document**

```markdown
The supervisor runs every isolated batch serially, writes `supervisor_manifest.json`,
and emits merged artifacts only after all batches pass strict coverage validation.
```

- [ ] **Step 2: Run focused tests and full non-GPU test suite**

Run: `conda run --no-capture-output -n tailguardkv-base pytest -q tests/test_diagnostic_batch_runner.py`

- [ ] **Step 3: Launch the supervisor asynchronously under the existing lock convention**

```bash
nohup setsid flock -n out/locks/diagnostic_session27_gpu01.lock \
  env CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output --cwd "$PWD" -n tailguardkv-base \
  python scripts/run_diagnostic_session_batches.py ... \
  > out/logs/diagnostic_session27_supervisor_YYYYMMDD_HHMMSS.nohup.log 2>&1 < /dev/null &
```

- [ ] **Step 4: After process completion, verify manifest, all batch status records, and merged artifact row count**

Run: `conda run --no-capture-output -n tailguardkv-base python -c '...'`
