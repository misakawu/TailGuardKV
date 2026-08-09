from __future__ import annotations

from unittest.mock import patch

from profiles.full import FullKVAdapter
from run_util.core_types import ProfileMeasurement, Request


def test_full_adapter_uses_qwen2_exact_runtime_for_measured_batches() -> None:
    adapter = FullKVAdapter({"pilot_model": "/tmp/model"})
    requests = [Request("r0", "qa", "hello", session_id="s1", turn_index=0)]
    measurement = ProfileMeasurement(
        request_id="r0",
        profile="full_gpu",
        adapter="full",
        ok=True,
        measured=True,
        output_text="ok",
        latency_ms=1.0,
        ttft_ms=1.0,
        peak_memory_mib=1.0,
        kv_cache_memory_mib=1.0,
        resident_memory_mib=1.0,
    )

    with patch("profiles.full.qwen2_exact_profile_many_measurements", return_value=[measurement]) as qwen2_many:
        rows = adapter.profile_many(requests, "full_gpu", dry_run=False, session_runtime={}, memory_budget_mib=64.0)

    assert rows == [measurement]
    qwen2_many.assert_called_once()
