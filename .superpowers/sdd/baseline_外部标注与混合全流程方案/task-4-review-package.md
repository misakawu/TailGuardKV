## Task 4 Review Package

### Scope

- Reviewed only Task 4's authorized shared labeling implementation:
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/build_hybrid_sessions.py`
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts/select_hybrid_candidate_batches.py`
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/tests/test_hybrid_sessions.py`
- No source files were edited.

### Spec Verdict

**FAIL.** The candidate builder, profile-loss classifier, and async configuration guard are substantially implemented, but the required final-form reserve/export and source-provenance controls are bypassable.

### Quality Verdict

**FAIL.** The focused test suite passes, but it asserts the invalid eight-per-cell final export behavior and lacks adversarial coverage for duplicate source records, unproven RAGhot fallback, false exhaustion state, and malformed final role/provenance metadata.

### Findings

1. **P1** `scripts/label_sharegpt_sessions.py:62`, `:172`, `:44`: final fixture export requires only 8 candidates per risk/task cell and never invokes the 12-reserve selector.
2. **P1** `scripts/build_hybrid_sessions.py:337`; `scripts/label_sharegpt_sessions.py:291`: uniqueness uses `(dataset, request_id)` instead of `content_source_record_id`, so reused source records can reach final output.
3. **P1** `scripts/select_hybrid_candidate_batches.py:65`; `scripts/build_hybrid_sessions.py:33`: a Boolean/CLI flag enables RAGhot QA without verified LongBench QA exhaustion and strict underfill.
4. **P1** `scripts/select_hybrid_candidate_batches.py:76`, `:210`: construction errors are reported as `source_exhausted=True` even when the supplied state is false.
5. **P2** `scripts/label_sharegpt_sessions.py:224`, `:282`: final validation omits `content_source_record_id`, `hybrid_turn_role`, the required five-turn role sequence, and cross-turn task consistency.

### Checks

- `python3 -m pytest tests/test_hybrid_sessions.py` completed successfully: `17 passed in 0.31s`.
- The test result does not clear the findings because the current tests do not exercise the bypass paths above.
