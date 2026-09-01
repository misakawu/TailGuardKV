# Baseline Session Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a 48-session hybrid `baseline_session` fixture with independently reported backend and risk-signal gates.

**Architecture:** The external labeling workspace builds and re-measures deterministic five-turn ShareGPT/LongBench hybrid sessions. The repository validates their provenance and evaluates two independent gates; a passing trace gate always permits backend policy replay, while a failed risk gate labels the resulting policy comparison `risk_evidence_insufficient`.

**Tech Stack:** Python 3.10+, pytest, existing Qwen2.5 profile runner, JSONL/CSV artifacts.

## Global Constraints

- Use exactly 48 sessions and 240 requests: 16 each of `kivi_sensitive`, `h2o_sensitive`, and `low_risk`/`tie_sensitive`.
- Each session has five consecutive turns; `turn2` is QA or Summary, and `turn3`/`turn4` use the same LongBench reference.
- Calibration/eval are session-level and each risk group/task combination has four sessions on each side.
- Session risk labels must come from a hybrid-session re-measurement, never from a LongBench single-turn label.
- A KIVI or H2O sensitive label requires a family maximum of at least `0.05` and a cross-family gap of at least `0.02`; ties never fill a sensitive group.
- `trace_semantics_gate` and `risk_signal_gate` are separate JSON artifacts. A failed risk gate must not suppress backend replay.

---

### Task 1: Build And Label Hybrid Session Fixtures

**Files:**
- Create: `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/build_hybrid_sessions.py`
- Modify: `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`
- Test: `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_hybrid_sessions.py`

**Interfaces:**
- Produces a JSONL fixture where every row has `task`, `reference`, `session_id`, `turn_index`, `arrival_index`, and hybrid provenance metadata.
- `label_sharegpt_sessions.build_fixture(rows, measurements)` consumes hybrid re-measurements and returns the interleaved fixture plus manifest.

- [ ] **Step 1: Write failing fixture-builder tests** for 48 five-turn sessions, provenance fields, `turn2` QA/Summary semantics, repeated reference on turns 3/4, and interleaved arrival indices.
- [ ] **Step 2: Run the focused external test** and confirm it fails because the hybrid builder does not exist.
- [ ] **Step 3: Implement the builder**: retain ShareGPT turns 0/1, assign LongBench content to turn 2, derive deterministic recall/rewrite prompts for turns 3/4, and retain `content_source_dataset`, request id/index, template, and original session id in metadata.
- [ ] **Step 4: Write failing risk-pool and split tests** for strict thresholds, tie exclusion, insufficient-pool errors, and four calibration/four eval sessions for each risk-group/task combination.
- [ ] **Step 5: Update labeling and run focused tests** so it re-measures the completed hybrid sessions, classifies only strict groups, splits by session, and assigns globally interleaved arrivals.
- [ ] **Step 6: Commit repository-side fixture contract changes only**; report external-workspace file changes explicitly because that workspace is not a Git repository.

### Task 2: Validate Hybrid Provenance And Risk Gate

**Files:**
- Modify: `scripts/import_external_fixtures.py`
- Modify: `scripts/validate_trace_quality.py`
- Modify: `tests/test_external_fixture_import.py`
- Modify: `tests/test_pilot_session_trace_dataset.py`
- Create: `out` gate JSON through normal execution only

**Interfaces:**
- `validate_baseline_session_fixture(path)` rejects malformed hybrid provenance and chat rows used as risk samples.
- `validate_trace_quality(measurements, fixture_rows)` returns a JSON-serializable risk-gate result including group means, task coverage, provenance failures, and `passed`.

- [ ] **Step 1: Write failing repository tests** for required hybrid metadata, rejected missing LongBench provenance, QA/Summary-only risk records, strict per-family signal, positive sensitive-vs-control gap, and provenance traceability.
- [ ] **Step 2: Run the focused tests** and confirm they fail against the legacy fixture contract/gate.
- [ ] **Step 3: Extend fixture validation and risk evaluation minimally** using the exact global thresholds and session-level split counts.
- [ ] **Step 4: Serialize the risk-gate JSON** with explicit errors and sufficient evidence to trace each quality record back to its injected content and template.
- [ ] **Step 5: Run the focused tests and the existing session gate tests.**
- [ ] **Step 6: Commit the repository changes.**

### Task 3: Separate Backend Replay From Risk Evidence

**Files:**
- Modify: `run_experiment.py`
- Modify: `run_util/experiment_summary.py`
- Modify: `configs/pilot_external_baseline_session.yaml`
- Modify: `tests/test_session_trace_pressure.py`
- Modify: `tests/test_external_baseline_configs.py`

**Interfaces:**
- The baseline-session command writes `trace_semantics_gate` and `risk_signal_gate` JSON paths configured under `outputs`.
- Policy summary rows contain `policy_comparison_status`, set to `risk_evidence_insufficient` only when trace semantics passed but risk evidence failed.

- [ ] **Step 1: Write failing runner tests** proving trace failure blocks replay, risk failure does not block replay, and policy results are marked non-comparable.
- [ ] **Step 2: Run the focused tests** and confirm the legacy runner blocks policy replay on a risk-gate failure.
- [ ] **Step 3: Split the runner control flow**: enforce the trace semantics gate before replay, run replay after it passes, then attach the independent risk-gate result and comparison status.
- [ ] **Step 4: Add configured JSON output paths and propagate comparison status into CSV/JSON summaries.**
- [ ] **Step 5: Run focused tests, all session-related tests, and config validation.**
- [ ] **Step 6: Commit the repository changes.**

### Task 4: Generate, Import, And Run The Corrected Experiment

**Files:**
- Modify: `docs/status/2026-08-12_baseline_smoke_execution_log.md`
- Generate: external hybrid fixture/manifest/measurements and timestamped `out/` run outputs

**Interfaces:**
- The imported fixture passes `scripts/import_external_fixtures.py --validate-only`.
- The experiment reports both gates independently and preserves backend output when risk evidence is insufficient.

- [ ] **Step 1: Generate the hybrid fixture and run its measured profile pass in the external labeling workspace.**
- [ ] **Step 2: Import the generated fixture using validate-only, then the normal import command.**
- [ ] **Step 3: Run `bash scripts/run_external_baseline_smoke.sh session` and wait for completion.**
- [ ] **Step 4: Inspect both gate JSON files, policy summary status, trace evidence, and output paths; append factual results to the execution log.**
- [ ] **Step 5: Run the complete repository test suite and commit repository documentation/configuration changes.**
