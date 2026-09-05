# Task 3 Report: Separate Backend Replay From Risk Evidence

## Status

Task 3 implementation is complete in the five owned repository files. The baseline-session runner now treats `trace_semantics_gate` as the sole replay blocker. `risk_signal_gate` is evaluated only after policy replay and cannot suppress backend output. Risk failure is reported as `risk_evidence_insufficient`; risk success is reported as `formally_comparable`.

No real profile run, smoke run, external fixture import, or generated experiment artifact was executed.

## Implemented Behavior

- Added explicit `smoke_trace_semantics_gate` and `smoke_risk_signal_gate` output paths to `configs/pilot_external_baseline_session.yaml`.
- Added trace semantics validation for fixture identity/order, profile completeness, resident KV/history fields, session reuse, global resident evolution, and a real pressure event. This gate is serialized as JSON and blocks policy replay when it fails.
- Kept split validation non-blocking for replay; a failed split contributes to insufficient risk evidence while preserving all policy outputs once trace semantics pass.
- Moved risk-signal evaluation after all configured policy sweeps. Risk exceptions are serialized as `evaluation_error` and still produce `risk_evidence_insufficient`.
- Propagated `policy_comparison_status` into runner JSON, each policy-run payload/metrics, ordinary summary CSV rows, and total-policy summary CSV rows.
- On trace-gate failure, writes a separate risk JSON with `status: not_evaluated`, propagates `policy_comparison_status: not_evaluated` into the runner payload and experiment summary row, and does not invoke either risk evaluation or policy replay.

## TDD Evidence

### RED

Added execution-order and propagation assertions to `tests/test_session_trace_pressure.py`. Before the runner change, the focused branch tests failed because the observed order was `risk -> policy` and trace failure still invoked the risk gate. The failure also exposed that a mocked risk result could not be serialized on the trace-failure path.

### GREEN

After moving risk evaluation below policy replay and making the trace-failure risk artifact explicit, the same focused branch tests passed. The tests now assert:

- `stage_order == ["policy", "risk"]` when trace semantics pass.
- `run_policies` and `validate_trace_quality` are both untouched when trace semantics fail.
- Trace-gate failure propagates `not_evaluated` to the top-level runner payload and experiment summary row.
- `risk_evidence_insufficient` appears in risk JSON, nested policy metrics, ordinary summary CSV, and total-policy summary CSV.
- `formally_comparable` appears on the corresponding successful path.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_session_trace_pressure.py tests/test_external_baseline_configs.py` — **11 passed**.
- `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/*session*.py tests/test_external_baseline_configs.py` — **73 passed**.
- YAML output-path validation for `configs/pilot_external_baseline_session.yaml` — **ok**.
- `python -m py_compile run_util/experiment.py run_util/experiment_summary.py tests/test_session_trace_pressure.py tests/test_external_baseline_configs.py` — **ok**.
- Scoped `git diff --check` for the five owned files — **ok**.

## Commit

The five owned files are committed with:

`baseline_session门禁与回放解耦`

Trace-failure status propagation is covered by follow-up commit `5f79bad` (`fix: propagate trace gate comparison status`).

## Concerns

- Existing unrelated user deletions, TODO files, `docs/superpowers/`, and generated `out/` directories remain untouched and uncommitted.
- The checked-in external fixture is still governed by Task 4's hybrid fixture generation/import workflow; this task only changes runner control flow and summaries.
