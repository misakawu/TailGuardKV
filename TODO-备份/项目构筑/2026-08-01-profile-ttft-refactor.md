# First-Token TTFT Profile Refactor Plan

Goal: make formal profile runs produce real TTFT values using explicit first-token timing, and fail when required TTFT is missing.

Architecture: replace one-shot `model.generate()` timing with explicit prefill plus autoregressive decode. TTFT is measured from request start through the first emitted token. Full, KIVI, and H2O share the same semantics; KIVI/H2O keep their custom cache path.

Global constraints:
- Do not fill `ttft_ms` from `latency_ms`, `stage_generate_ms`, or total generation time.
- Use `ttft_semantics="first_token"` only when `ttft_ms` is measured from a real first-token boundary.
- `configs/pilot_50.yaml` formal profile runs must fail if measured rows lack TTFT.
- No need to preserve the old one-shot generate implementation.

Tasks:
1. Add `profiles/generation_timing.py` with `generate_with_first_token_timing(...)` and focused tests.
2. Replace runtime `model.generate(...)` paths in `profiles/transformers_runtime.py` and `profiles/qwen2_runtime_common.py`, and expose prefill/first-token extras in profile tables.
3. Extend validation/config wiring with `require_ttft`.
4. Run focused tests and async profile smoke/pilot verification.
