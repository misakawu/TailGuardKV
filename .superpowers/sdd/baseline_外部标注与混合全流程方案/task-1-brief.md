# Task 1: Source-agnostic provenance and quality gate

Read this first. It is the complete requirement for this task.

## Files in the isolated TailGuardKV worktree

- Modify `scripts/import_external_fixtures.py`.
- Modify `scripts/validate_trace_quality.py`.
- Modify `run_util/experiment.py`.
- Modify `tests/test_external_fixture_import.py`.
- Add `tests/test_baseline_quality_gate.py`.

## Requirements

1. Replace hard-coded validation of `sharegpt_longbench_hybrid_session`, `longbench`, and `longbench_content` with a source registry.
2. Session content sources may be `longbench` or `raghot_qa`; the top-level construction source remains `hybrid_session_builder`.
3. Fixed five-turn roles are `sharegpt_opening`, `sharegpt_opening`, `content_query`, `reference_recall`, and `reference_rewrite` for turns 0 through 4.
4. Every injected turn must contain `content_source_dataset`, `content_source_request_id`, `content_source_index`, `content_payload_hash`, `injection_template`, `original_session_id`, and `hybrid_turn_role`.
5. RAGhot rows additionally require `context_pack_hash`, `supporting_fact_ids`, and `packing_policy_version`.
6. Add `baseline_quality_signal_gate` JSON output. It checks a 180-row strict fixture with three 60-row risk groups; calibration/eval coverage; complete measurement for all eight profiles; positive sensitive-vs-low-risk evidence; and at least one KIVI and one H2O profile whose group mean `quality_loss` exceeds 0.02.
7. Set `policy_comparison_status` to `formally_comparable` only when the corresponding signal gate passes. Otherwise it must be `risk_evidence_insufficient` while preserving policy output.

## Required tests

- Legal LongBench and RAGhot provenance passes.
- Missing RAGhot context/evidence, forged source, mismatched role/turn, incomplete profile coverage, and relabeled ties fail.
- A failed quality gate keeps policy output and returns `risk_evidence_insufficient`.

## Global constraints

- Model and profiles are fixed: `full_gpu`, four KIVI profiles, and three H2O profiles.
- Final labels must be derived from complete final-form input measurement; pre-screening labels cannot be final-fixture risk fields.
- Sensitive threshold is 0.05; family gap is at least 0.02; low-risk losses are at most 0.01; ties are rejected.
- Follow test-driven development: add a focused failing test, run it to observe the expected failure, implement the smallest change, then run the relevant test suite.

## Report contract

Write the detailed report to `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/task-1-report.md`: files changed, exact test commands/results, commits, and remaining concerns. Return only status (`DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`), commit IDs, one-line test summary, and concerns.
