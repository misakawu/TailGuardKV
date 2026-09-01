# Task 3 Report: Separate Backend Replay From Risk Evidence

## Status

Task 3 is implemented and verified. No real model profile, backend smoke, or
production experiment was started. `baseline_quality` behavior and files were
left unchanged.

## Implemented behavior

- `run_util/experiment.py` now creates an explicit `trace_semantics_gate`
  JSON for `baseline_session`. It checks fixture request/session/turn/arrival
  structure, measured profile completeness, resident KV fields, accumulated
  history, and a real measured-replay probe for session reuse, global resident
  evolution, and at least one pressure event.
- The trace gate is the only gate that can stop policy replay. A failed trace
  writes `risk_signal_gate.status=not_evaluated` and returns before replay.
- Policy replay runs before risk evaluation. A failed independent risk gate
  preserves all backend policy outputs and sets
  `policy_comparison_status=risk_evidence_insufficient`; a passing risk gate
  uses `formally_comparable`.
- `run_util/experiment_summary.py` propagates comparison status into
  experiment rows, policy rows, and total-policy summary rows.
- `configs/pilot_external_baseline_session.yaml` declares separate
  `smoke_trace_semantics_gate` and `smoke_risk_signal_gate` JSON paths.
- Legacy runner tests were updated with a valid synthetic pressure trace and
  explicit coverage for trace blocking, risk non-blocking, independent gate
  JSON, and comparable/non-comparable summary statuses.

## TDD evidence

### RED

Before the implementation, the focused runner/config command reproduced the
legacy behavior:

```text
3 failed, 7 passed, 1 error
```

The existing runner returned `2` when `validate_trace_quality` failed and had
no `validate_trace_semantics` or separate gate output. The config test also
failed because the two new output paths did not exist.

### GREEN

Focused command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_session_trace_pressure.py \
  tests/test_external_baseline_configs.py
```

Result: `11 passed`.

Session/config/risk/backend regression command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_session_trace_pressure.py \
  tests/test_pilot_session_trace_dataset.py \
  tests/test_external_fixture_import.py \
  tests/test_external_baseline_configs.py \
  tests/test_experiment_semantics.py \
  tests/test_policy_session_budget.py \
  tests/test_session_refactor.py \
  tests/test_session_summary_visuals.py \
  tests/test_build_session_profile_table.py \
  tests/test_session_profile_payloads.py
```

Result: `88 passed`.

Summary regression subset:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_tailguard_core.py \
  -k 'summary_rows or total_policy_summary_rows or pilot_summary_contains_each_policy_sweep_row'
```

Result: `4 passed, 202 deselected`.

Syntax and whitespace checks passed:

```bash
python -m py_compile run_util/experiment.py run_util/experiment_summary.py \
  tests/test_session_trace_pressure.py tests/test_external_baseline_configs.py
git diff --check -- run_util/experiment.py run_util/experiment_summary.py \
  configs/pilot_external_baseline_session.yaml \
  tests/test_session_trace_pressure.py tests/test_external_baseline_configs.py
```

An independent diff review found no remaining Critical or Important issues and
verified trace → replay → risk/status ordering and all required status paths.

## Changed files

- `run_util/experiment.py`
- `run_util/experiment_summary.py`
- `configs/pilot_external_baseline_session.yaml`
- `tests/test_session_trace_pressure.py`
- `tests/test_external_baseline_configs.py`
- `.superpowers/sdd/2026-09-01-baseline-session-correction/task-3-report.md`

Only these six Task 3 files are included in the Task 3 commit. Existing dirty
documentation moves, generated `out/` directories, and unrelated files remain
untouched and uncommitted.
