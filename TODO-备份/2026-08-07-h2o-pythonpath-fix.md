# H2O Pythonpath Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure H2O measured profiles launch `profiles.qwen2_kv_runtime` with the required `third_party/H2O/h2o_hf` path in `PYTHONPATH`, then restart the pilot experiment with `nohup`.

**Architecture:** Keep the fix in the shared Qwen2 KV runtime launcher so adapter-specific path requirements can be forwarded without duplicating subprocess code. Cover the single-request helper and the H2O batch adapter path with focused regression tests, then relaunch the existing experiment entrypoint.

**Tech Stack:** Python, pytest/unittest, conda, nohup

## Global Constraints

- Keep the change minimal and localized to the existing Qwen2 KV runtime launch path.
- Use TDD: write the failing regression test first, run it red, then implement the fix.
- Do not alter unrelated user changes in the worktree.
- Put the `nohup` log in `out/`, which is the sibling directory of `out/profile_tables/`.

---

### Task 1: Add Regression Coverage For H2O Pythonpath Propagation

**Files:**
- Modify: `tests/test_tailguard_core.py`

**Interfaces:**
- Consumes: `qwen2_kv_profile_measurement(adapter, env_name, request, spec, runtime_config, timeout_s=None, pythonpath=(), extra=None) -> ProfileMeasurement`
- Produces: Regression tests that require single-request runtime launches and H2O batch adapter launches to forward `pythonpath`

- [ ] Add a failing test that calls `qwen2_kv_profile_measurement(..., pythonpath=("third_party/H2O/h2o_hf",))` and asserts the spawned subprocess environment includes that path in `PYTHONPATH`.
- [ ] Run `pytest tests/test_tailguard_core.py::TailGuardCoreTest::test_qwen2_profile_measurement_appends_pythonpath -v` and confirm it fails because the helper does not yet support or propagate the extra path.
- [ ] Tighten the existing H2O batch adapter test so it asserts `pythonpath=adapter.pythonpath` is forwarded into `qwen2_kv_profile_many_measurements`.

### Task 2: Implement The Minimal Runtime Fix

**Files:**
- Modify: `profiles/base.py`
- Modify: `profiles/h2o.py`

**Interfaces:**
- Consumes: Existing H2O adapter `pythonpath` tuple and shared Qwen2 KV runtime launcher
- Produces: Shared single-request helper support for `pythonpath`, plus H2O adapter calls that pass `self.pythonpath` to both single and batch measured runtime helpers

- [ ] Extend `qwen2_kv_profile_measurement()` to accept a `pythonpath: Sequence[str] = ()` keyword argument and append absolute versions of those paths ahead of any pre-existing `PYTHONPATH`.
- [ ] Update `H2OAdapter.profile()` and `H2OAdapter.profile_many()` so measured runs pass `pythonpath=self.pythonpath`.
- [ ] Keep the batch helper behavior unchanged except for receiving the already-supported `pythonpath` argument from the adapter.

### Task 3: Verify And Relaunch

**Files:**
- Modify: `out/pilot_measured.pid` if the process is relaunched
- Create/append: `out/pilot_measured.nohup.log`

**Interfaces:**
- Consumes: Updated tests and experiment entrypoint `python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml`
- Produces: Passing regression tests and a fresh background experiment process with PID/log artifacts

- [ ] Run the focused regression tests for the new single-request helper coverage and the H2O adapter batch coverage.
- [ ] Start the experiment with `nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml > out/pilot_measured.nohup.log 2>&1 < /dev/null & echo $! > out/pilot_measured.pid`.
- [ ] Verify the PID exists and the log file is created without reporting an immediate startup failure.
