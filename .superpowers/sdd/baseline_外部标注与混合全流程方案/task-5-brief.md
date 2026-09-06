# Task 5: Export strict quality and session fixtures

Read this first. It is the complete requirement.

## Authorized shared labeling files

- Modify `../TailGuardKV-labeling/scripts/label_longbench_quality.py`.
- Modify `../TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`.
- Add `../TailGuardKV-labeling/tests/test_strict_fixture_selection.py`.

The user authorized direct changes in this non-Git directory. Do not change unrelated files or initialize Git.

## baseline_quality

1. Remove all `tie_sensitive` refill behavior. Ties go only to a rejected manifest.
2. Export `baseline_quality_external.jsonl` with 60 strict rows in each KIVI-sensitive, H2O-sensitive, and low-risk group: 180 total, split 90 calibration / 90 eval.
3. LongBench is primary. RAGhot QA can fill a quality risk family only when measured LongBench strict QA is insufficient.
4. Manifest records source distribution, task distribution, rejected ties, complete eight-profile coverage, and fixture hash.

## baseline_session

1. Export `baseline_session_external.jsonl` with 48 five-turn sessions / 240 rows.
2. Each `risk x task x split` cell has exactly 4 sessions; each risk/task has 8 sessions.
3. Sort by risk margin descending then source request ID ascending; assign alternating calibration/eval in that order.
4. Manifest records candidate reserve, source, original skeleton ID, content ID, payload hash, and every profile's coverage.

## Tests

- Quality: strict three groups of 60 and 90/90 split.
- Session: 48x5 and all twelve risk/task/split cells have 4.
- Reject ties, duplicate content, duplicate final skeleton, missing provenance, and mismatched source hash.

## Constraints

- Final labels derive only from complete final-form eight-profile measurements.
- Thresholds: sensitive >= 0.05, family gap >= 0.02, low-risk <= 0.01; ties rejected.
- RAGhot is QA-only and only after verified LongBench exhaustion/insufficiency.
- Use TDD. No GPU campaign in this implementation task; async tests/programs with logs and no polling.

## Report

Write results to `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/task-5-report.md`, including files, tests, and concerns. No Git commit is possible in target directory. Return short status only.
