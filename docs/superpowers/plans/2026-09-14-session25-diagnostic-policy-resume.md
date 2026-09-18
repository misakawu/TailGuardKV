# Session25 Diagnostic Policy Resume Implementation Plan

> **For agentic workers:** Implement sequentially in the requested worktree; do not commit without explicit user approval.

**Goal:** Produce validated diagnostic-only baseline smoke policy tables and TTFT/KV/event visualizations from existing 25-session measurements.

**Architecture:** Validate and merge all existing profile batches, then reuse the current policy grid runner and per-seed aggregation. Keep original measurement and backup directories intact.

**Tech Stack:** Python, pytest, existing Conda environment `tailguardkv-base`, CSV/JSON, matplotlib.

## Global Constraints

- Do not treat `diagnostic_only=true` as a formal baseline result.
- Launch all tests and experiments asynchronously with `nohup` + `setsid`; do not poll while running.
- Stop on incomplete measurement coverage or failed policy cells; retain logs and exit codes.

---

### Task 1: Recover complete profile batches

**Files:** `scripts/run_diagnostic_session_batches.py`, `tests/test_diagnostic_batch_runner.py`

- [ ] Test recognition of retry-generated profile CSV names and rejection of ambiguous files.
- [ ] Recognize exactly one eligible CSV per batch and write a successful supervisor manifest after `--validate-existing` merges.
- [ ] Run focused tests, then `--validate-existing`; require 1000 validated rows and `merged=true` before continuing.

### Task 2: Run diagnostic policy grid

**Files:** `scripts/run_session25_fourth_sweeps.py`, `out/full_25session_baseline_9_14/full/`

- [ ] Start the existing sweep asynchronously only after Task 1 succeeds.
- [ ] Verify the complete grid status, per-seed raw policy tables, and diagnostic provenance when the user reports completion.

### Task 3: Audit visual outputs

**Files:** `scripts/aggregate_session27_baselines.py`, `out/full_25session_baseline_9_14/full/seed*/policy_tables/`

- [ ] Confirm each seed's `baseline_smoke.md`, total summary, events, session points, and TTFT/KV plots exist.
- [ ] Reconcile raw row counts and aggregate values; report any discrepancy without inventing data.
