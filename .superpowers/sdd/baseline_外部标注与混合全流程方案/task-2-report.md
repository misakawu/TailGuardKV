# Task 2 report: verifiable RAGhot QA candidates

## Status

Complete. The authorized non-Git labeling workspace now creates formal RAGhot QA candidates only when every supporting fact resolves to a nonempty sentence under the same context title. It packs supporting sentences in original context order before distractors, preserves the 3,000-character ceiling without truncating evidence, and records deterministic provenance hashes.

## Changed paths

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/common.py`
  - Added the RAGhot parquet source and candidate-output constants.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/prepare_raghot_qa_candidates.py`
  - Added the parquet CLI and pure candidate builder.
  - Exports QA-only rows with `source_record_id`, `supporting_fact_ids`, `context_pack_hash`, `payload_hash`, `content_payload_hash`, and `packing_policy_version=raghot_support_first_v1` in `metadata`.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_raghot_candidates.py`
  - Added coverage for full evidence/provenance, rejection paths, and support-first ordering.

The labeling directory is not a Git repository. No commit is possible or was created.

## TDD and verification

| Command | Result |
| --- | --- |
| `pytest -q tests/test_raghot_candidates.py` before implementation | RED: collection failed with `ModuleNotFoundError: No module named 'prepare_raghot_qa_candidates'`, confirming the requested builder did not yet exist. |
| `pytest -q tests/test_raghot_candidates.py` after implementation | GREEN: `3 passed in 0.03s`. |
| `python3 scripts/prepare_raghot_qa_candidates.py --output /tmp/tailguardkv-raghot-qa-candidates.jsonl` | Exit 0; produced and independently validated 2,088 rows, all `task=qa`, with nonempty evidence IDs and 64-character SHA-256 provenance hashes. |
| `pytest -q` | `14 passed in 0.30s`. |

All script/test processes were launched asynchronously with preserved logs under `/tmp/tailguardkv-task2-*.log`.

## Concerns

None. The CLI intentionally wrote its real-data validation artifact to `/tmp` rather than the default shared artifact path, keeping this task within its explicitly authorized source/test/common-file scope.

## Fix round 1/5

### Changed paths

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/prepare_raghot_qa_candidates.py`
  - Validates `answer` by its normalized nonempty form while retaining the raw parquet string as the exported `reference` and as the payload-hash input.
  - Defines the fixed 3,000-character maximum and rejects invalid limits both before CLI parquet reads and in `build_candidates`.
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_raghot_candidates.py`
  - Covers whitespace preservation in both `reference` and the deterministic payload hash.
  - Covers rejection of `max_context_chars=3001`.

No commit was created: `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling` is not a Git repository.

### Verification

| Command | Result |
| --- | --- |
| `pytest -q tests/test_raghot_candidates.py` before the fix | RED: `2 failed, 3 passed`; it showed the whitespace-stripped reference and missing upper-bound exception. |
| `pytest -q tests/test_raghot_candidates.py` after the fix | GREEN: `5 passed in 0.01s`. |
| `pytest -q` | `16 passed in 0.24s`. |

All test processes were launched asynchronously with preserved logs at `/tmp/tailguardkv-task2-fix1-*.log`.
