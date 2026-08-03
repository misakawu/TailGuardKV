# Task 3 Report

## Scope

- Added `run_util/mem_test.py` as the independent `run_mem_test` orchestrator.
- Added `run_mem_test.py` CLI wrapper.
- Added the Task 3 orchestration regression test in `tests/test_tailguard_core.py`.
- Confirmed `configs/budget_sweep_small.yaml` is absent in this worktree, so no deletion was needed.
- Left `configs/pilot.yaml` unchanged.

## TDD record

1. Added the failing orchestration test first.
2. Ran:
   ```bash
   pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
   ```
3. Observed the expected failure on August 3, 2026:
   - `ImportError: cannot import name 'mem_test' from 'run_util'`
4. Implemented the orchestrator and CLI wrapper.
5. Re-ran the same focused command and confirmed pass.

## Implementation notes

- `run_util.mem_test.run()` now:
  - builds the budget series with `build_budget_series()`
  - generates a derived config with `build_mem_test_config()`
  - writes `configs/run_mem_test.generated.yaml` under the selected run directory
  - runs measured profile collection with `formal_run=False`, `dry_run=False`, `import_measurements=""`, and `allow_import_measurements_for_debug=False`
  - validates measured profile rows with `require_measured=True`
  - sweeps the Cartesian product of `epsilons x deltas x budgets`
  - forces `allow_dry_run_replay=False`
  - writes summary CSVs, total summary CSV, analysis JSON/Markdown, and visualization outputs
- The orchestrator returns non-zero only for execution failures. Analysis with `found_passing_budget=false` does not flip the exit code.
- Output path resolution strips a leading `out/` and anchors generated artifacts directly under the requested `run_dir`, which avoids nested `run_dir/out/...` and `run_dir/run_dir/...` mistakes.

## Verification

Focused Task 3 red test:

```bash
pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
```

- Before implementation: failed with the expected `ImportError`
- After implementation: passed

Related Task 1/2 regression set:

```bash
pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_budget_series_uses_100_mib_steps tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_generated_config_removes_tailguard_and_uses_relative_outputs tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_finds_budget_with_kv_and_ttft_gain tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_reports_no_budget_when_ttft_not_better tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_records_ttft_gain_but_kv_miss_as_near_miss tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
```

- Result: `6 passed`

## Small adjustment made during verification

- The brief’s new test placed filesystem existence assertions after the `TemporaryDirectory()` context exited, which would remove the run directory before those assertions ran.
- I kept the asserted values unchanged and moved those assertions inside the temporary-directory scope so the test checks the intended behavior instead of the cleanup side effect.

## Commit plan

- Stage only:
  - `run_util/mem_test.py`
  - `run_mem_test.py`
  - `tests/test_tailguard_core.py`
- Commit message:
  - `feat: add run_mem_test launcher`

## Fix round 1

### Review finding

- `run_mem_test` reused `write_summary()` without overriding the experiment-row label, so `run_mem_test_summary.csv` incorrectly emitted `pilot-smoke-measured` as the experiment name.

### TDD record

1. Extended `test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs` to assert:
   - the summary experiment row name is `run_mem_test`
   - every policy invocation uses `allow_dry_run_replay=False`
2. Ran:
   ```bash
   pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
   ```
3. Observed the expected failure on August 3, 2026:
   - summary row `name` was `pilot-smoke-measured` instead of `run_mem_test`
4. Implemented the minimal fix:
   - `experiment_summary.summary_rows()` now accepts `payload["experiment_name"]` with the existing `pilot-smoke-measured` string preserved as the default
   - `run_util.mem_test` now sets `experiment_name="run_mem_test"` on every summary payload it writes
5. Re-ran the focused test and the related mem_test regression set.

### Files changed in fix round 1

- `experiment_summary.py`
- `run_util/mem_test.py`
- `tests/test_tailguard_core.py`

### Verification

Focused test:

```bash
pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
```

- Result: `1 passed`

Related mem_test regression set:

```bash
pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_budget_series_uses_100_mib_steps tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_generated_config_removes_tailguard_and_uses_relative_outputs tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_finds_budget_with_kv_and_ttft_gain tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_reports_no_budget_when_ttft_not_better tests/test_tailguard_core.py::TailGuardCoreTest::test_mem_test_analysis_records_ttft_gain_but_kv_miss_as_near_miss tests/test_tailguard_core.py::TailGuardCoreTest::test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs -q
```

- Result: `6 passed`

### Commit

- Commit message:
  - `fix: label mem test summaries`
