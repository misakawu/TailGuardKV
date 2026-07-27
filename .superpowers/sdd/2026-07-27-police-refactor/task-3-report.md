# Task 3 Report: Implement Independent Static Policies

## Status
Completed.

## What Changed
- `policies/static_best.py` now inherits directly from `Policy`, owns its own `PolicyContext`, and selects the global lossy profile with `mean_loss <= epsilon` and the lowest calibration p95 TTFT.
- `policies/static_safe.py` now inherits directly from `Policy`, owns its own `PolicyContext`, and selects the global lossy profile with `mean_loss <= epsilon`, `violation_rate <= delta`, and the lowest calibration p95 TTFT.
- Both static policies preserve the required action fields and fall back to `engine_full_lru` when no lossy profile qualifies.
- `tests/test_policy_refactor.py` now covers direct inheritance, selection rules, and fallback behavior.

## Verification
- `python -m pytest tests/test_policy_refactor.py -k static -v`
- `python -m pytest -v`

## Concerns
- None.
