# Task 2: Build verifiable RAGhot QA candidates

Read this first. It is the complete requirement for this task.

## Files in the explicitly authorized shared labeling directory

- Add `../TailGuardKV-labeling/scripts/prepare_raghot_qa_candidates.py`.
- Add `../TailGuardKV-labeling/tests/test_raghot_candidates.py`.
- Modify `../TailGuardKV-labeling/scripts/common.py` only as required.

The user explicitly authorized direct changes to this non-Git directory. Do not make unrelated changes there or in the main worktree.

## Input and filtering

Input is `/DATACENTER3/zhenxiang.wang/resource/RAGhot_QA/validation-00000-of-00001.parquet`.

1. Read `id`, `question`, `answer`, `context`, and `supporting_facts`.
2. Reject records whose ID is empty or whose question or answer is empty.
3. Each supporting fact must locate its `sent_id` in a context sentence with the same title. Reject the entire record if any fact cannot be located. Only fully locatable records are formal candidates.
4. Put every supporting sentence first in original document/sentence order; append nonsupporting sentences in the same original order as distractor context.
5. Context has a 3,000-character ceiling. Reject when all supporting sentences cannot fit; never truncate from the head.
6. The exact prompt is `Context + Question + Answer:` and the reference is the original `answer`.

## Output contract

Each candidate must have a stable request ID, `task=qa`, reference, source record ID, complete evidence ID list, context-pack SHA-256, payload SHA-256, and `packing_policy_version=raghot_support_first_v1`.

## Required tests

- Fully evidenced sample retains all facts and produces stable hashes.
- Missing supporting title/sentence, over-limit supporting text, and empty fields are rejected.
- Nonsupporting sentences appear only after every supporting sentence.

## Global constraints

- RAGhot is QA-only backup. It must never introduce Summary content.
- Provenance fields must support the Task 1 source registry: `raghot_qa` rows require `context_pack_hash`, `supporting_fact_ids`, and `packing_policy_version`.
- Use test-driven development: establish a focused failing test first, observe the expected failure, then implement the smallest change and run focused coverage.

## Report contract

Write the detailed report to `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/task-2-report.md`: files changed, exact test commands/results, and concerns. Because the target directory has no Git repository, state that no commit is possible and record the exact changed paths. Return only status, commit IDs if any, one-line test summary, and concerns.
