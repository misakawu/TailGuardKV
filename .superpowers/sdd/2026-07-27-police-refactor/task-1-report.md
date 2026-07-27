Task 1 completed: extracted shared policy helper code and updated the oracle TTFT import to use the common percentile helper.

Changes made:
- Added `policies/common.py` with:
  - `ProfileStats`
  - `percentile`
  - `profile_stats`
  - `PolicyContext`
- Updated `policies/quality_oracle.py` to import `percentile` from `policies.common`.
- Kept `quality_oracle` behavior aligned with the brief: it still uses evaluation truth and remains marked `oracle=True`.

Validation:
- `python -m pytest tests/test_policy_common.py -v`
- `python -m pytest -v`

Result:
- Focused policy-common tests passed.
- Full test suite passed: 58 tests.
