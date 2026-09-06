# Task 3: Extend LongBench and run direct pre-screening

Read this first. It is the complete requirement for this task.

## Authorized shared labeling files

- Modify `../TailGuardKV-labeling/scripts/prepare_longbench_candidates.py`.
- Modify `../TailGuardKV-labeling/configs/longbench_labeling.yaml`.
- Add `../TailGuardKV-labeling/scripts/partition_labeling_batches.py`.
- Add `../TailGuardKV-labeling/tests/test_labeling_batches.py`.
- Modify `../TailGuardKV-labeling/scripts/run_longbench_labeling.sh`.
- Modify `../TailGuardKV-labeling/scripts/label_longbench_quality.py`.

The user explicitly authorized direct edits in this non-Git directory. Do not initialize Git or edit unrelated paths.

## Requirements

1. Scan every available LongBench QA and Summary record. Do not limit either class to its first 200 rows.
2. Record LongBench subset, original source ID, task, reference, prompt hash, and candidate order.
3. Partition deterministically into 60-request batches. Each batch manifest lists input hashes, all eight expected profiles, and its corresponding CSV output.
4. Direct profile measurements are launched asynchronously and serially; merge only rows whose request ID, profile, and candidate hash agree and whose `ok=true` and `measured=true` values are present.
5. Use strict direct-risk ranking to prioritize later hybrid content. If capacity is insufficient, continue with the next LongBench batch until the hybrid prescreen need is met or LongBench is exhausted.

## Review-mandated fixes within approved expanded scope

1. Direct labels must never populate final-fixture risk fields, including through the legacy LongBench runner and quality exporter.
2. The generated async measurement command must execute with the TailGuardKV repository importable (for example by changing into that repository before `python -m run_util.build_profile_table`).
3. The executable batch flow must consume completed prescreens and a per-cell need so it stops when strict-risk capacity is sufficient or reports LongBench exhaustion.
4. GPU serialization must use an inter-process atomic lock/claim, not only a PID-file existence check.
5. The merger must validate actual CSV identity, including row task, against the candidate, even when the runner does not emit a `candidate_hash` column.

## Tests

- Identical input yields identical partitions and manifests.
- Merger rejects duplicate request/profile, mismatched candidate hash, failed profiles, and incomplete-profile CSVs.
- Direct labels appear only in prescreen manifests, never in final-fixture risk fields.

## Global constraints

- Eight profiles: `full_gpu`, four KIVI, and three H2O.
- Direct pre-screen labels may rank candidates but cannot be copied into final fixture risk fields.
- Any actual profiling process is GPU-exclusive and must use the plan's asynchronous `nohup + setsid`, Conda runner pattern, PID/log files, with no polling and no concurrent two-GPU job.
- Follow TDD: focused RED result before implementation; then minimal implementation and green focused coverage.

## Report contract

Write a full report to `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/task-3-report.md`: files changed, exact tests/commands/results, whether any real profile run was launched, PIDs/logs if so, and concerns. The target is non-Git: record exact changed paths and no commit. Return only status, commit IDs if any, one-line test summary, and concerns.
