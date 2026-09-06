# Task 4: Build, measure, and select hybrid session candidates

Read this first. It is the complete requirement for this task.

## Authorized shared labeling files

- Modify `../TailGuardKV-labeling/scripts/build_hybrid_sessions.py`.
- Modify `../TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`.
- Add `../TailGuardKV-labeling/scripts/select_hybrid_candidate_batches.py`.
- Modify `../TailGuardKV-labeling/tests/test_hybrid_sessions.py`.

The user authorized direct edits in this non-Git directory. Do not change unrelated files or initialize Git.

## Construction

1. Every candidate has two ShareGPT opening turns, a QA or Summary content turn, then fixed recall and rewrite turns.
2. Prefer LongBench QA. Use RAGhot QA only when every LongBench QA batch is complete and a strict QA cell remains below 12 complete hybrid candidates.
3. Summary is LongBench-only. If a Summary cell has fewer than 12, extend LongBench or report failure; never use RAGhot or ties.
4. Skeleton reuse is allowed only during exploration. Final fixture candidates require unique `original_session_id`, source record, and injected payload.
5. A measurement batch contains 12 sessions and 60 requests using the persistent-worker dual-GPU setup, launched asynchronously and serially.

## Classification and stopping

1. Classify each session using its maximum loss among turns 2--4 for each profile.
2. Retain at least 12 complete, strictly valid candidates in each `kivi_sensitive`, `h2o_sensitive`, `low_risk` by `qa`, `summary` cell.
3. When underfilled, construct another batch ordered by prescreen priority. If sources exhaust, output an insufficiency report with exact cell/count/exhaustion state and do not export a fixture.

## Tests

- Five-turn completeness, global arrival interleaving, and session-internal consistent provenance.
- RAGhot QA builds; RAGhot Summary is rejected.
- Final selection rejects duplicate skeleton, duplicate content, ties, and incomplete profile coverage.
- Any underfilled cell fails export and includes exact cell, available count, and source-exhausted state.

## Constraints

- Final labels use complete final-form hybrid measurement only, never direct pre-screen labels.
- Strict thresholds: sensitive >=0.05, cross-family gap >=0.02, low-risk <=0.01; ties rejected.
- The eight fixed profiles are full_gpu, four KIVI, three H2O.
- GPU commands must be async `nohup`/`setsid`, PID/logged, and two-GPU serial; do not launch a real campaign during this implementation task.
- Use TDD and record RED then green results.

## Report

Write full results to `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/task-4-report.md`, including changed paths, test results, any GPU launch, and concerns. The target is non-Git: no commit. Return only the short status contract.
