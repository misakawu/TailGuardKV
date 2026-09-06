# Task 3 report: LongBench full scan and direct prescreen batches

## Status

Complete. The authorized non-Git labeling workspace now scans every available configured LongBench QA and Summary record, assigns stable provenance/hash/order metadata, creates deterministic 60-request direct-prescreen batches, materializes per-batch runner configs, and strictly merges only complete, successful, hash-matching eight-profile measurements.

Direct risk labels are persisted only in batch `*.prescreen.json` manifests. The helper that hands candidates to later hybrid construction preserves `risk_family=unlabeled` and omits direct risk fields. The serial launcher builds the required `nohup setsid env CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base python -m run_util.build_profile_table ... --no-dry-run` command and launches at most one pending batch per invocation.

## Changed paths

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/prepare_longbench_candidates.py`
  - Removed the 200-per-task selection cap and scans all available configured QA/Summary records.
  - Records source subset/ID/index, task/reference, SHA-256 `prompt_hash`, global `candidate_order`, and deterministic `candidate_hash`.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/configs/longbench_labeling.yaml`
  - Defines 60-request direct-prescreen batches, all eight profiles, serial GPU exclusivity, and the required Conda/CUDA launcher contract.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/partition_labeling_batches.py`
  - Added deterministic batch manifests, per-batch config materialization, asynchronous serial launcher construction, strict CSV merger, strict direct-risk ranking, prescreen persistence, and hybrid-need scheduler.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_labeling_batches.py`
  - Added deterministic partition/manifest, runner/config binding, strict merge rejection, direct-label isolation, and scheduler coverage.

The labeling workspace is not a Git repository. No Git repository was initialized and no commit exists.

## TDD and verification

All test/program processes were launched asynchronously using `nohup setsid ... > LOG 2>&1 < /dev/null &`, with a PID file and a single shell `wait` completion signal; no polling was used.

| Stage | Command | Result | Log / PID |
| --- | --- | --- | --- |
| RED 1 | `pytest -q tests/test_labeling_batches.py` | Expected collection failure: `ModuleNotFoundError: partition_labeling_batches`. | `/tmp/tailguardkv-task3-red-20260902.log`, `.pid` |
| GREEN 1 | `pytest -q tests/test_labeling_batches.py` | `6 passed in 0.05s`. | `/tmp/tailguardkv-task3-green-20260902.log`, `.pid` |
| RED 2 | `pytest -q tests/test_labeling_batches.py` | Expected missing `write_batch_configs` import. | `/tmp/tailguardkv-task3-config-red-20260902.log`, `.pid` |
| GREEN 2 | `pytest -q tests/test_labeling_batches.py` | `7 passed in 0.10s`. | `/tmp/tailguardkv-task3-config-green-20260902.log`, `.pid` |
| RED 3 | `pytest -q tests/test_labeling_batches.py` | Expected missing `write_prescreen_manifest` import. | `/tmp/tailguardkv-task3-prescreen-red-20260902.log`, `.pid` |
| GREEN 3 | `pytest -q tests/test_labeling_batches.py` | `7 passed in 0.10s`. | `/tmp/tailguardkv-task3-prescreen-green-20260902.log`, `.pid` |
| RED 4 | `pytest -q tests/test_labeling_batches.py` | Expected missing `next_batch_for_hybrid_need` import. | `/tmp/tailguardkv-task3-scheduler-red-20260902.log`, `.pid` |
| GREEN 4 | `pytest -q tests/test_labeling_batches.py` | `8 passed in 0.05s`. | `/tmp/tailguardkv-task3-scheduler-green-20260902.log`, `.pid` |
| Real-data non-GPU check | `python3 scripts/prepare_longbench_candidates.py --output /tmp/tailguardkv-task3-longbench-candidates.jsonl --manifest /tmp/tailguardkv-task3-longbench-manifest.json` | Exit 0; generated 1,950 all-source QA/Summary candidates, proving the former 200-per-class cap is absent. | `/tmp/tailguardkv-task3-prepare-20260902.log`, `.pid` |
| Batch integration non-GPU check | `python3 scripts/partition_labeling_batches.py --candidates /tmp/tailguardkv-task3-longbench-candidates.jsonl --output-root /tmp/tailguardkv-task3-prescreen-batches --base-config configs/longbench_labeling.yaml` | Exit 0; generated 33 deterministic batch/config/manifest sets. | `/tmp/tailguardkv-task3-partition-20260902.log`, `.pid` |
| Final suite | `pytest -q` | `24 passed in 0.31s`. | `/tmp/tailguardkv-task3-final-pytest-20260902.log`, `.pid` |
| Compile | `python3 -m py_compile scripts/prepare_longbench_candidates.py scripts/partition_labeling_batches.py` | Exit 0. | `/tmp/tailguardkv-task3-pycompile-20260902.log`, `.pid` |

## Profile execution

No real profile/GPU measurement was launched, as required for Task 3. Therefore there are no profile PIDs or profile logs. The PID/log files above belong only to short-lived local tests, compilation, and non-GPU candidate/manifest generation.

## Concerns

The existing `scripts/run_longbench_labeling.sh` is outside Task 3's authorized paths and was intentionally not changed. Operational stages must invoke `partition_labeling_batches.py` with `--base-config` and then call `--launch-next` only after the previous GPU job has completed; Task 6 owns the ordered actual campaign.

## Fix round 1/5

### Changed paths

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/run_longbench_labeling.sh`
  - Removed the direct profile-table-to-`label_longbench_quality.py` workflow.
  - Now prepares candidates and invokes the batch CLI with its prescreen directory, per-cell requirement, and one asynchronous `--launch-next` attempt.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/label_longbench_quality.py`
  - Refuses to create any final quality fixture from direct prescreen measurements. A future final fixture may only be produced from the fully shaped, final-input workflow in Task 5.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/partition_labeling_batches.py`
  - Generated profile commands now `cd /DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline` before importing `run_util`.
  - Replaced the PID observation check with a nonblocking `flock` held by the detached GPU runner for its full lifetime. A competing invocation cannot obtain the lock and cannot overlap a two-GPU job.
  - Added `--completed-prescreens-dir` and `--required-per-cell`; CLI reports `pending`, `need_met`, or `longbench_exhausted`, and launches only the selected next batch while capacity remains insufficient.
  - CSV hash fallback now requires emitted `task` to exactly equal the candidate task before hashing prompt/reference.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_labeling_batches.py`
  - Added coverage for repo import cwd and atomic lock contract, direct-export prohibition, fallback task mismatch, and completed-prescreen CLI stopping behavior.

### TDD and verification

All commands were launched asynchronously using `nohup setsid ... > LOG 2>&1 < /dev/null &`, each with a PID file and one `wait` completion signal. No command polled a live process.

| Stage | Command | Result | Log / PID |
| --- | --- | --- | --- |
| RED | `pytest -q tests/test_labeling_batches.py` | Expected failures: missing repo `cd`/`flock` command contract, direct exporter did not reject, fallback accepted task mismatch, CLI rejected completed-prescreen arguments. | `/tmp/tailguardkv-task3-fix1-red-20260903.log`, `.pid` |
| GREEN | `pytest -q tests/test_labeling_batches.py` | `11 passed in 0.29s`. | `/tmp/tailguardkv-task3-fix1-green-20260903.log`, `.pid` |
| Full suite | `pytest -q` | `27 passed in 0.44s`. | `/tmp/tailguardkv-task3-fix1-full-pytest-20260903.log`, `.pid` |
| Syntax | `python3 -m py_compile scripts/partition_labeling_batches.py scripts/label_longbench_quality.py && bash -n scripts/run_longbench_labeling.sh` | Exit 0. | `/tmp/tailguardkv-task3-fix1-syntax-20260903.log`, `.pid` |

### Profile execution and concern

No real profile/GPU measurement was launched. Therefore no profile PID/log exists; the PID/log files above belong only to local tests and syntax checks. The generated runner uses `flock -n` as an atomic claim. A second concurrent invoker can start only a lock-contending lightweight launcher that exits before invoking Conda/GPU work; it cannot overlap the active two-GPU profile job.
