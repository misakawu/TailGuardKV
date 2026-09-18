# Session25 Policy Runtime Minimal Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a fail-closed diagnostic GPU gate for exact execution, KIVI multi-turn cache reuse, and full→KIVI→full runtime switching before the full policy grid.

**Architecture:** A standalone Python script derives the actual evaluation split from the merged profile table, selects stable complete sessions, and executes three isolated scenarios through `OnlineQwenSessionBackend`. It writes one provenance-rich JSON report and returns nonzero on any runtime, TTFT, reuse, or transition failure.

**Tech Stack:** Python, pytest, existing `tailguardkv-base` Conda environment, JSON, existing TailGuardKV backend/profile APIs.

## Global Constraints

- All outputs remain `diagnostic_only=true` and cannot support formal paper conclusions.
- CPU tests use `nohup + setsid` and may be polled because the user explicitly allowed test polling.
- GPU validation uses `nohup + setsid`; do not poll while it runs, and wait for the user to report completion.
- Do not overwrite prior failed policy CSVs, logs, status files, or profile measurements.
- Do not start the full 240-policy grid unless this validation succeeds.

---

### Task 1: Build Evaluation Session Selection and Report Gate

**Files:**
- Create: `scripts/validate_session25_policy_runtime.py`
- Create: `tests/test_session25_policy_runtime_validation.py`

**Interfaces:**
- Consumes: config path, run root, merged profile CSV, fixture requests, `split_measurements(...)`.
- Produces: `select_validation_sessions(config: dict, measurements: list[ProfileMeasurement], requests: list[Request]) -> list[list[Request]]` and `validate_scenario(name: str, rows: list[dict[str, object]], requirements: dict[str, object]) -> dict[str, object]`.

- [ ] Write tests proving selection uses only evaluation requests, returns three stable complete sessions, and rejects insufficient session coverage.
- [ ] Run the focused test asynchronously and verify RED because the new module does not exist.
- [ ] Implement split-aware selection using configured `split_seed`, `calibration_fraction`, and `stratify_session`.
- [ ] Write tests proving invalid TTFT, failed runtime rows, missing KIVI reuse, and missing transition/recompute evidence fail closed.
- [ ] Implement compact scenario validation and JSON-safe result formatting.
- [ ] Run focused tests asynchronously and require all PASS.

### Task 2: Execute Three Isolated Backend Scenarios

**Files:**
- Modify: `scripts/validate_session25_policy_runtime.py`
- Modify: `tests/test_session25_policy_runtime_validation.py`

**Interfaces:**
- Consumes: `OnlineQwenSessionBackend`, `build_profile_adapters`, `Action`, selected evaluation sessions.
- Produces: `run_validation(config_path: Path, run_root: Path, output_path: Path) -> int`.

- [ ] Write backend-factory tests for exact two-turn, KIVI three-turn, and full→KIVI→full execution order; assert every backend closes in success and failure paths.
- [ ] Run focused tests asynchronously and verify RED.
- [ ] Implement exact scenario with `full_gpu` turns 0–1 and valid TTFT checks.
- [ ] Implement KIVI scenario with `kivi_4bit_residual64` turns 0–2 and reuse checks on turns 1–2.
- [ ] Implement switch scenario with full turn 0, KIVI turn 0 from another selected session, then full turn 1; require runtime transition or recompute evidence on the final request.
- [ ] Write the report atomically after all scenarios or after the first failed scenario, retaining raw error text and selected request provenance.
- [ ] Run focused and related regression tests asynchronously; require clean PASS output.

### Task 3: Launch Diagnostic GPU Gate

**Files:**
- Modify: `完整实验进度.md`
- Create at runtime: `out/full_25session_baseline_9_14/logs/minimal_policy_runtime_20260915/`

**Interfaces:**
- Consumes: completed script and existing merged profile/fixture/config.
- Produces: `validation.json`, `validation.log`, `validation.pid`, and `validation.exit`.

- [ ] Run `git diff --check` and related CPU tests before GPU launch.
- [ ] Start the validation with `setsid nohup conda run --no-capture-output -n tailguardkv-base python scripts/validate_session25_policy_runtime.py --config configs/pilot_diagnostic_session25_fourth.yaml --run-root out/full_25session_baseline_9_14/full --output out/full_25session_baseline_9_14/logs/minimal_policy_runtime_20260915/validation.json`.
- [ ] Record PID immediately; redirect stdout/stderr to `validation.log` and write the process return code to `validation.exit`.
- [ ] Do not poll the GPU validation. Wait for the user to report completion.
- [ ] After user notification, inspect the exit code, JSON report, and log; update `完整实验进度.md` with exact success or failure evidence.
