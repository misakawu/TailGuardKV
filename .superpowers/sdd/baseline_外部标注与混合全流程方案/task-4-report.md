# Task 4 report: hybrid session construction, reserve selection, and batch interface

## Status

Complete. The authorized non-Git labeling workspace can construct five-turn hybrid sessions with session-consistent provenance, use RAGhot only as an explicit QA fallback, retain strict fully measured 12-session reserves per risk/task cell, and build a 12-session/60-request asynchronous serial measurement interface. No final fixture is exported by the new candidate-batch selector.

## Changed paths

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/build_hybrid_sessions.py`
  - Adds explicit RAGhot QA inputs and rejects any RAGhot Summary content.
  - Supports task-specific 12-session batches while retaining balanced defaults.
  - Carries a single content source/request/payload provenance value through all five turns, including required RAGhot evidence provenance.
  - Uses canonical `content_query`, `reference_recall`, and `reference_rewrite` turn roles.
  - Can enforce unique final skeleton, content source, and injected payload identities.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`
  - Requires complete coverage of all eight profiles, including `full_gpu`, for final-form hybrid classification.
  - Adds strict candidate-reserve selection with exact insufficiency messages containing cell, available count, required count, and source-exhausted state.
  - Rejects duplicate skeleton/content/payload provenance and session-internal provenance inconsistency.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/select_hybrid_candidate_batches.py`
  - New selector for a single 12-session/60-request next batch, with LongBench-first QA and RAGhot QA only after LongBench QA exhaustion.
  - Returns complete, pending-measurement, or exact insufficient status without exporting a fixture.
  - Builds a `nohup setsid flock -n` CUDA `0,1` serial-launch command with PID/log paths.
  - Validates a persistent-worker, one-profile-at-a-time, dual-GPU config before a real launch.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_hybrid_sessions.py`
  - Adds RAGhot QA/Summary, canonical provenance role, eight-profile, reserve-underfill/duplicate/coverage, batch command, and config-gate coverage.

## TDD and verification

All local test and compile processes were launched asynchronously with `nohup setsid ... > LOG 2>&1 < /dev/null &`, with PID files and a single `wait` completion signal. No process was polled.

| Stage | Result | Log / PID |
| --- | --- | --- |
| RED 1 | Expected collection failure: missing `select_hybrid_candidate_reserves`. | `/tmp/tailguardkv-task4-red-20260903.log`, `.pid` |
| GREEN 1 | Focused hybrid coverage: `16 passed in 0.33s`. | `/tmp/tailguardkv-task4-green2-20260903.log`, `.pid` |
| RED 2 | Expected collection failure: missing `validate_hybrid_measurement_config`. | `/tmp/tailguardkv-task4-config-red-20260903.log`, `.pid` |
| GREEN 2 | Focused hybrid coverage: `17 passed in 0.30s`; syntax compile exited 0. | `/tmp/tailguardkv-task4-config-green-20260903.log`, `/tmp/tailguardkv-task4-pycompile2-20260903.log`, `.pid` |
| RED 3 | Expected assertion: legacy `longbench_content` role differed from required `content_query`. | `/tmp/tailguardkv-task4-provenance-red-20260903.log`, `.pid` |
| GREEN 3 | Focused hybrid coverage: `17 passed in 0.34s`. | `/tmp/tailguardkv-task4-provenance-green-20260903.log`, `.pid` |
| Final suite | `33 passed in 0.51s`. | `/tmp/tailguardkv-task4-final-20260903.log`, `.pid` |
| Final syntax | `python3 -m py_compile` for all three modified/new scripts exited 0. | `/tmp/tailguardkv-task4-final-pycompile-20260903.log`, `.pid` |

## GPU launch

No real profile/GPU measurement was launched. The new command construction and pre-launch persistent-worker dual-GPU config gate were exercised only in tests. No GPU PID or GPU log exists.

## Concerns

The pre-existing `configs/hybrid_session_candidates.yaml` currently has `use_persistent_workers: false` and `profile_chunk_size: 4`; it is outside Task 4's authorized paths. The new launcher deliberately rejects it until the later operational/config task supplies a compliant persistent-worker configuration. The labeling directory is non-Git; no Git repository was initialized and no commit was created.

## Fix round 1/5

### Review fixes

- `build_fixture` now requires 12 complete, strict, uniquely provenanced final-form candidates in every risk/task cell before selecting the fixed 8-per-cell export. A 48-session candidate set alone can no longer export a fixture.
- Final uniqueness keys use `(content_source_dataset, content_source_record_id)`, not request ID. The source record ID is required provenance.
- RAGhot QA fallback now requires a structured Task 3 exhaustion report with `status=longbench_exhausted` and scheduler counts. The removed boolean CLI switch could not prove exhaustion. Underfill is checked from the current strict final-form counts before fallback is considered.
- Candidate-construction insufficiency keeps the caller-provided actual `source_exhausted` state; a local lack of currently supplied rows no longer fabricates exhaustion.
- Final selection/export validates all five canonical roles, `last_turn`, session-consistent source/content/payload provenance, and the required RAGhot evidence metadata on every RAGhot turn.

### Regression tests and verification

All commands used the same asynchronous `nohup setsid` + PID/log + single `wait` pattern; no process was polled.

| Stage | Result | Log / PID |
| --- | --- | --- |
| RED | Expected failures covering the 8-candidate export bypass, request-ID uniqueness, unverified RAGhot flag, false exhaustion report, and missing session-role validation. | `/tmp/tailguardkv-task4-fix1-red-20260903.log`, `.pid` |
| GREEN | Focused hybrid suite: `21 passed in 0.47s`. | `/tmp/tailguardkv-task4-fix1-green2-20260903.log`, `.pid` |
| Full suite | `37 passed in 0.57s`. | `/tmp/tailguardkv-task4-fix1-full-20260903.log`, `.pid` |
| Syntax | `python3 -m py_compile` for the three Task 4 scripts exited 0. | `/tmp/tailguardkv-task4-fix1-pycompile-20260903.log`, `.pid` |

No real GPU/profile measurement was launched in this fix round.

After making `content_source_record_id` individually mandatory, the final fresh verification again reported `37 passed in 0.57s` and a successful three-script compile: `/tmp/tailguardkv-task4-fix1-final-20260903.log`, `/tmp/tailguardkv-task4-fix1-final-pycompile-20260903.log` (with matching `.pid` files).

## Fix round 2/5

### Review fixes

- The hybrid selector no longer accepts caller-provided strict counts or a LongBench exhaustion Boolean/report. It derives strict final-form cell counts from the existing hybrid candidates plus complete eight-profile measurements.
- RAGhot QA eligibility now requires every locally supplied LongBench manifest containing QA records to have a matching completed prescreen with exactly the same candidate and ranked request-ID sets. The selector reads manifest/prescreen directories directly; it does not trust a status string.
- `build_hybrid_sessions.py` no longer exposes the `--use-raghot-qa` or `--raghot` CLI switches. RAGhot construction remains an internal selector route after manifest verification.
- Final hybrid validation requires turns 2--4 to have exactly one shared allowed task (`qa` or `summary`) and one shared reference, and rejects RAGhot content if that shared task is Summary.

### Regression tests and verification

| Stage | Result | Log / PID |
| --- | --- | --- |
| RED | Expected failures for the old selector signature/authorization path and missing final content task/reference validation. | `/tmp/tailguardkv-task4-fix2-red-20260903.log`, `.pid` |
| GREEN | Focused hybrid suite: `23 passed in 0.52s`. | `/tmp/tailguardkv-task4-fix2-green-20260903.log`, `.pid` |
| Final suite | `39 passed in 0.71s`. | `/tmp/tailguardkv-task4-fix2-final-20260903.log`, `.pid` |
| Syntax | `python3 -m py_compile` for all Task 4 scripts exited 0. | `/tmp/tailguardkv-task4-fix2-pycompile-20260903.log`, `.pid` |

No real GPU/profile measurement was launched in this fix round. The persistent-worker configuration concern remains unchanged.

## Fix round 3/5

### Review fixes

- RAGhot authorization now compares the union of QA request IDs in every supplied LongBench batch manifest against a separate authoritative LongBench candidate inventory. Any omitted, duplicate, or uncompleted expected QA batch rejects fallback.
- The same verifier checks each completed prescreen's candidate and ranked request-ID sets against its manifest, then derives remaining QA underfill from fully measured final-form hybrid candidates.
- `build_hybrid_sessions(..., use_raghot_qa=True)` now invokes that verifier itself. It requires authoritative rows, manifests, completed prescreens, and actual hybrid measurement state; direct programmatic RAGhot construction without them raises an error.
- The selector passes the same evidence through to the builder, and its CLI defaults the authoritative inventory to the complete LongBench candidate source while keeping it separately configurable for explicit reproducible inventories.

### Regression tests and verification

| Stage | Result | Log / PID |
| --- | --- | --- |
| RED | With the authoritative QA-inventory equality check temporarily removed, a partial manifest incorrectly selected RAGhot; the regression failed exactly as expected. | `/tmp/tailguardkv-task4-fix3-red-20260903.log`, `.pid` |
| GREEN | Focused hybrid suite: `25 passed in 0.53s`. | `/tmp/tailguardkv-task4-fix3-green-20260903.log`, `.pid` |
| Final suite | `41 passed in 0.63s`. | `/tmp/tailguardkv-task4-fix3-final-20260903.log`, `.pid` |
| Syntax | `python3 -m py_compile` for all Task 4 scripts exited 0. | `/tmp/tailguardkv-task4-fix3-pycompile-20260903.log`, `.pid` |

No real GPU/profile measurement was launched in this fix round. The persistent-worker configuration concern remains unchanged.

## Independent review (2026-09-03)

**Verdict: FAIL.** The focused suite passes (`17 passed in 0.31s`), and the builder correctly creates five-turn globally interleaved candidates, rejects RAGhot Summary input, computes each profile's maximum over turns 2--4, checks all eight fixed profiles, and validates the persistent dual-GPU configuration. However, the end-to-end final-form contract is not enforced on the export path.

### Findings

1. **P1 - The fixture export bypasses the required 12-candidate reserves.** `build_fixture()` calls `select_balanced_sessions()` directly, whose fixed threshold is eight sessions per risk/task cell. `main()` then writes that fixture directly. `select_hybrid_candidate_reserves(..., required_per_cell=12)` is not called by either path, so a final fixture can be exported from exactly eight valid candidates per cell. The existing test explicitly encodes that eight-per-cell behavior. This violates the required 12 complete final-form candidates per cell and the no-export-on-underfill requirement. See `scripts/label_sharegpt_sessions.py:62`, `scripts/label_sharegpt_sessions.py:172`, and `scripts/label_sharegpt_sessions.py:44`.
2. **P1 - Final source uniqueness is checked against request ID, not source record ID.** The contract requires unique original skeleton, source record, and injected payload. Both builder and final validation use `(content_source_dataset, content_source_request_id)` as their source uniqueness key, while `content_source_record_id` is neither required by final validation nor used in the uniqueness set. Two separately generated requests for the same source record therefore pass final selection. See `scripts/build_hybrid_sessions.py:337` and `scripts/label_sharegpt_sessions.py:291`.
3. **P1 - RAGhot fallback can be activated without proving LongBench QA exhaustion.** The selector treats the caller-provided Boolean `longbench_qa_exhausted` as proof and the builder CLI exposes `--use-raghot-qa` with no exhaustion or underfill guard at all. A caller can therefore select RAGhot QA while LongBench QA candidates remain, contrary to the LongBench-first requirement. See `scripts/select_hybrid_candidate_batches.py:65` and `scripts/build_hybrid_sessions.py:33`.
4. **P1 - Insufficiency reporting can falsely claim source exhaustion.** When candidate construction raises for any reason, the selector returns `_status_report("insufficient", ...)`; `_status_report` always emits `source_exhausted=True` for that status, even if the actual `source_exhausted` input was false. This breaks the exact exhaustion-state reporting requirement and can misdirect the orchestrator. See `scripts/select_hybrid_candidate_batches.py:76` and `scripts/select_hybrid_candidate_batches.py:210`.
5. **P2 - Final selection does not validate the complete role/provenance contract.** Candidate validation omits `content_source_record_id` and `hybrid_turn_role`; it also does not enforce the required role sequence or QA/Summary consistency across turns 2--4. Mutated or malformed five-turn candidates can pass final-form measurement classification as long as the limited metadata fields remain present. See `scripts/label_sharegpt_sessions.py:224` and `scripts/label_sharegpt_sessions.py:282`.

### Verification

- Asynchronously launched focused review command: `python3 -m pytest tests/test_hybrid_sessions.py`
- Result: `17 passed in 0.31s`
- Log: `/tmp/tailguardkv-task4-review-pytest.log`
- PID record: `/tmp/tailguardkv-task4-review-pytest.pid`
