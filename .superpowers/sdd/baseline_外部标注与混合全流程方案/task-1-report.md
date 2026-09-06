# Task 1 Report: Source-Agnostic Provenance and Quality Gate

## Delivered

- Added a source registry for `longbench` and `raghot_qa` hybrid-session provenance.
- Enforced the fixed five-turn roles and the injected-turn payload hash.
- Enforced RAGhot context, supporting-fact, and packing-policy provenance.
- Added `baseline_quality_signal_gate` with strict 180-row, 60-per-family, 90/90 split, eight-profile, final-form label, threshold, gap, low-risk, and tie checks.
- Connected the baseline-quality gate to JSON output and policy comparison status without suppressing policy output.

## Files Changed

- `scripts/import_external_fixtures.py`
- `scripts/validate_trace_quality.py`
- `run_util/experiment.py`
- `tests/test_external_fixture_import.py`
- `tests/test_baseline_quality_gate.py`
- `tests/test_pilot_session_trace_dataset.py` (updated an existing fixture helper to the new role/hash contract)

## Test Evidence

| Command | Result |
| --- | --- |
| `pytest -q tests/test_external_fixture_import.py tests/test_baseline_quality_gate.py` | Initial RED: 11 failed, 6 passed; new API and provenance contract were absent. |
| `pytest -q tests/test_external_fixture_import.py tests/test_baseline_quality_gate.py` | PASS: 17 passed in 1.38s. |
| `pytest -q tests/test_pilot_session_trace_dataset.py tests/test_session_trace_pressure.py` | PASS: 27 passed in 1.51s. |
| `pytest -q` | 352 passed, 3 subtests passed; 4 unrelated failures listed below. |

All pytest commands were launched with `nohup setsid`; logs and PID files are under `out/task-1-logs/` and are ignored by Git.

## Commit

- `e1901f1 feat: 增加 baseline 质量信号门禁`

## Remaining Concerns

- Full-suite failures outside Task 1:
  - `tests/test_sharegpt_labeling_strategy.py::test_build_fixture_allows_low_risk_only_sessions`
  - `tests/test_sharegpt_labeling_strategy.py::test_build_fixture_interleaves_sessions_in_pairs`
  - `tests/test_tailguard_core.py::TailGuardCoreTest::test_pilot_configs_use_full_gpu_only_formal_grid`
  - `tests/test_tailguard_core.py::TailGuardCoreTest::test_qwen2_profile_measurement_appends_pythonpath`
- The present imported external fixtures are legacy/non-strict data and are expected to fail the new gate until Tasks 2-5 produce validated final-form fixtures.

## Fix Round 1: Empty RAGhot Evidence

### Change

- `scripts/validate_trace_quality.py` now rejects empty list, tuple, set, and mapping values for required provenance fields, including RAGhot `supporting_fact_ids`.
- `tests/test_pilot_session_trace_dataset.py` adds a gate-path regression with otherwise valid RAGhot provenance and `supporting_fact_ids=[]`.

### Test Evidence

| Command | Result |
| --- | --- |
| `pytest -q tests/test_pilot_session_trace_dataset.py -k empty_raghot_supporting_fact_ids` | RED: 1 failed; the quality gate incorrectly passed empty evidence. |
| `pytest -q tests/test_pilot_session_trace_dataset.py tests/test_external_fixture_import.py` | PASS: 34 passed in 0.49s. |

### Commit

- `32d1d4b fix: 拒绝空 RAGhot 证据`
