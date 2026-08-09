from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from backends.measured_replay import MeasuredReplayBackend
from metrics import MetricCollector
from policies.tailguard import TailGuardPolicy
from run_util.core_types import BackendResult, CacheState, DeviceState, PolicyRunRecord, ProfileMeasurement, Request
from run_util.validation import validate_backend_results
from run_util.data_utils import load_requests
import run_util.data_utils as data_utils_module
import visual.plot_summary as plot_summary_module


def _measurement(
    request_id: str,
    profile: str,
    *,
    session_id: str | None = None,
    turn_index: int = 0,
    split: str = "calibration",
    quality_loss: float = 0.0,
    kv_incremental_mib: float | None = None,
    kv_cumulative_mib: float | None = None,
    resident_before: float | None = None,
    resident_after: float | None = None,
    restore_ms: float | None = None,
    recompute_ms: float | None = None,
    evicted_kv_mib: float | None = None,
    budget_hit: bool = False,
    ttft_ms: float = 10.0,
) -> ProfileMeasurement:
    total_kv = kv_cumulative_mib if kv_cumulative_mib is not None else 100.0
    return ProfileMeasurement(
        request_id=request_id,
        session_id=session_id,
        turn_index=turn_index,
        profile=profile,
        adapter="full",
        ok=True,
        measured=True,
        output_text=f"{request_id}:{profile}",
        latency_ms=ttft_ms + 1.0,
        ttft_ms=ttft_ms,
        peak_memory_mib=total_kv,
        kv_cache_memory_mib=total_kv,
        resident_memory_mib=resident_after if resident_after is not None else total_kv,
        quality_loss=quality_loss,
        kv_incremental_mib=kv_incremental_mib,
        kv_cumulative_mib=kv_cumulative_mib,
        resident_kv_mib_before=resident_before,
        resident_kv_mib_after=resident_after,
        restore_ms=restore_ms,
        recompute_ms=recompute_ms,
        evicted_kv_mib=evicted_kv_mib,
        budget_hit=budget_hit,
        extra={"task": "qa", "length_bucket": "short", "split": split},
    )


def test_load_requests_sharegpt_keeps_session_order_and_session_level_splits() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "sharegpt.json"
        data_path.write_text(
            json.dumps(
                [
                    {
                        "id": "sess-a",
                        "conversations": [
                            {"from": "human", "value": "Hello"},
                            {"from": "gpt", "value": "Hi"},
                            {"from": "human", "value": "Explain KV cache"},
                            {"from": "gpt", "value": "It stores attention history"},
                        ],
                    },
                    {
                        "id": "sess-b",
                        "conversations": [
                            {"from": "human", "value": "Summarize this"},
                            {"from": "gpt", "value": "Summary"},
                        ],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        requests, fallback = load_requests(
            {
                "data": {
                    "source": "sharegpt",
                    "requests": str(data_path),
                    "calibration_fraction": 0.5,
                }
            }
        )

    assert fallback is False
    assert [(request.session_id, request.turn_index) for request in requests] == [
        ("sess-a", 0),
        ("sess-a", 1),
        ("sess-b", 0),
    ]
    assert [request.request_id for request in requests] == [
        "sess-a_turn_000",
        "sess-a_turn_001",
        "sess-b_turn_000",
    ]
    assert requests[0].effective_prompt == "Hello"
    assert requests[1].history_turns == ("User: Hello", "Assistant: Hi")
    assert requests[1].effective_prompt == "User: Hello\nAssistant: Hi\nUser: Explain KV cache"
    assert [request.metadata["split"] for request in requests] == ["calibration", "calibration", "eval"]


def test_load_requests_sharegpt_filters_turns_over_configured_token_limit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "sharegpt.json"
        data_path.write_text(
            json.dumps(
                [
                    {
                        "id": "sess-a",
                        "conversations": [
                            {"from": "human", "value": "short prompt"},
                            {"from": "gpt", "value": "short reply"},
                            {"from": "human", "value": "very long prompt"},
                        ],
                    },
                    {
                        "id": "sess-b",
                        "conversations": [
                            {"from": "human", "value": "keep me"},
                        ],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with patch.object(
            data_utils_module,
            "_request_prompt_token_count",
            side_effect=lambda request, tokenizer: 1200 if request.request_id == "sess-a_turn_001" else 128,
        ), patch.object(data_utils_module, "_load_prompt_tokenizer", return_value=object()):
            requests, fallback = load_requests(
                {
                    "model": {"pilot_model": "/tmp/model", "cache_dir": "/tmp/cache"},
                    "profile_smoke": {"vllm_max_model_len": 1024, "local_files_only": True},
                    "data": {
                        "source": "sharegpt",
                        "requests": str(data_path),
                        "calibration_fraction": 0.5,
                    },
                }
            )

    assert fallback is False
    assert [request.request_id for request in requests] == ["sess-a_turn_000", "sess-b_turn_000"]
    assert [request.metadata["split"] for request in requests] == ["calibration", "eval"]


def test_profile_measurement_session_fields_round_trip_and_legacy_parse() -> None:
    measurement = _measurement(
        "sess-a_turn_001",
        "kivi_4bit",
        session_id="sess-a",
        turn_index=1,
        kv_incremental_mib=30.0,
        kv_cumulative_mib=70.0,
        resident_before=40.0,
        resident_after=60.0,
        restore_ms=4.0,
        recompute_ms=0.0,
        evicted_kv_mib=10.0,
        budget_hit=True,
    )

    parsed = ProfileMeasurement.from_row(measurement.to_row())
    legacy = ProfileMeasurement.from_row(
        {
            "request_id": "legacy",
            "profile": "full_gpu",
            "adapter": "full",
            "ok": "True",
            "measured": "True",
            "output_text": "x",
            "latency_ms": "2",
            "ttft_ms": "1",
            "peak_memory_mib": "3",
            "kv_cache_memory_mib": "3",
            "resident_memory_mib": "3",
            "quality_loss": "0",
            "task": "qa",
            "length_bucket": "short",
            "split": "eval",
        }
    )

    assert parsed.session_id == "sess-a"
    assert parsed.turn_index == 1
    assert parsed.kv_incremental_mib == 30.0
    assert parsed.kv_cumulative_mib == 70.0
    assert parsed.resident_kv_mib_before == 40.0
    assert parsed.resident_kv_mib_after == 60.0
    assert parsed.restore_ms == 4.0
    assert parsed.recompute_ms == 0.0
    assert parsed.evicted_kv_mib == 10.0
    assert parsed.budget_hit is True
    assert legacy.session_id is None
    assert legacy.turn_index == 0
    assert legacy.kv_cumulative_mib is None
    assert legacy.budget_hit is False


def test_measured_replay_backend_keys_by_session_turn_and_updates_session_cache() -> None:
    backend = MeasuredReplayBackend(
        [
            _measurement("sess-a_turn_000", "kivi", session_id="sess-a", turn_index=0, kv_incremental_mib=20.0, kv_cumulative_mib=20.0),
            _measurement("sess-a_turn_001", "kivi", session_id="sess-a", turn_index=1, kv_incremental_mib=25.0, kv_cumulative_mib=45.0),
            _measurement("sess-a_turn_001", "full_cpu", session_id="sess-a", turn_index=1, kv_incremental_mib=40.0, kv_cumulative_mib=80.0),
        ]
    )
    requests = [
        Request("sess-a_turn_000", "qa", "turn0", session_id="sess-a", turn_index=0),
        Request("sess-a_turn_001", "qa", "turn1", session_id="sess-a", turn_index=1),
    ]

    rows = backend.run(requests, ["kivi", "kivi"])

    assert all(isinstance(row, BackendResult) for row in rows)
    assert [row.kv_cumulative_mib for row in rows] == [20.0, 45.0]
    assert backend.cache_state.get_current_profile("sess-a") == "kivi"
    assert backend.cache_state.get_cumulative_kv("sess-a", "kivi") == 45.0

    switched = backend.run([requests[1]], ["full_cpu"])[0]

    assert switched.kv_cumulative_mib == 80.0
    assert backend.cache_state.get_current_profile("sess-a") == "full_cpu"
    assert backend.cache_state.get_cumulative_kv("sess-a", "full_cpu") == 80.0
    assert switched.replay_source == "measured_profile_table"


def test_measured_replay_emits_budget_resolution_event_fields() -> None:
    backend = MeasuredReplayBackend(
        [
            _measurement("s1_t0", "kivi", session_id="s1", turn_index=0, kv_incremental_mib=30.0, kv_cumulative_mib=30.0, resident_before=0.0, resident_after=30.0),
            _measurement("s2_t0", "kivi", session_id="s2", turn_index=0, kv_incremental_mib=30.0, kv_cumulative_mib=30.0, resident_before=0.0, resident_after=30.0),
        ],
        global_budget_mib=50.0,
    )

    rows = backend.run(
        [
            Request("s1_t0", "qa", "turn0", session_id="s1", turn_index=0, arrival_index=0),
            Request("s2_t0", "qa", "turn0", session_id="s2", turn_index=0, arrival_index=1),
        ],
        ["kivi", "kivi"],
    )

    assert rows[1].budget_hit is True
    assert rows[1].extra["event_kind"] == "evict"
    assert rows[1].extra["event_reason"] == "global_budget_pressure"
    assert rows[1].extra["victim_session"] == "s1"
    assert rows[1].extra["budget_resolution"] == "evict_other_sessions"


def test_policy_record_from_backend_result_uses_backend_runtime_fields() -> None:
    backend_result = BackendResult(
        request_id="sess-a_turn_001",
        session_id="sess-a",
        turn_index=1,
        profile="full_cpu",
        ok=True,
        measured=True,
        latency_ms=21.0,
        ttft_ms=20.0,
        peak_memory_mib=90.0,
        kv_cache_memory_mib=70.0,
        resident_memory_mib=60.0,
        kv_cumulative_mib=120.0,
        kv_incremental_mib=40.0,
        resident_kv_mib_before=50.0,
        resident_kv_mib_after=60.0,
        restore_ms=4.0,
        recompute_ms=6.0,
        evicted_kv_mib=10.0,
        budget_hit=False,
        quality_loss=0.02,
    )

    record = PolicyRunRecord.from_action_and_backend_result(
        policy_name="tailguard",
        request=Request("sess-a_turn_001", "qa", "turn1", session_id="sess-a", turn_index=1, metadata={"length_bucket": "short"}),
        action_profile="full_cpu",
        action_reason="fallback to exact",
        placeholder=False,
        exact_profiles={"full_cpu"},
        oracle=False,
        backend_result=backend_result,
        budget_hit=True,
    )

    assert record.action_profile == "full_cpu"
    assert record.reason == "fallback to exact"
    assert record.ttft_ms == 20.0
    assert record.kv_cache_memory_mib == 70.0
    assert record.restore_ms == 4.0
    assert record.budget_hit is True


def test_validate_backend_results_does_not_require_profile_sampling_fields() -> None:
    results = [
        BackendResult(
            request_id="sess-a_turn_001",
            session_id="sess-a",
            turn_index=1,
            profile="kivi",
            ok=True,
            measured=True,
            latency_ms=11.0,
            peak_memory_mib=40.0,
            kv_cache_memory_mib=30.0,
        )
    ]

    validate_backend_results(results, require_memory=True)


def test_tailguard_budget_filter_uses_session_cumulative_kv_and_exact_fallback() -> None:
    policy = TailGuardPolicy(
        calibration_measurements=[
            _measurement("sess-a_turn_000", "kivi", session_id="sess-a", turn_index=0, quality_loss=0.01, kv_cumulative_mib=20.0, ttft_ms=5.0),
            _measurement("sess-a_turn_000", "full_cpu", session_id="sess-a", turn_index=0, quality_loss=0.0, kv_cumulative_mib=20.0, ttft_ms=12.0),
        ],
        profiles=["kivi", "full_cpu"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_cpu"},
        memory_budget_mib=50.0,
    )
    request = Request("sess-a_turn_001", "qa", "next", session_id="sess-a", turn_index=1)
    cache_state = CacheState().with_session_turn(
        "sess-a",
        profile="kivi",
        turn_index=0,
        cumulative_kv_mib=60.0,
        resident_kv_mib=60.0,
    )

    action = policy.decide(request, cache_state, DeviceState())

    assert action.profile == "full_cpu"
    assert action.fallback_reason == "no calibrated safe lossy profile within memory budget"
    assert action.safe is True
    assert action.budget_hit is True


def test_metric_collector_policy_summary_includes_session_aggregates() -> None:
    records = [
        PolicyRunRecord(
            policy="tailguard",
            request_id="sess-a_turn_000",
            session_id="sess-a",
            turn_index=0,
            action_profile="kivi",
            ok=True,
            measured=True,
            task="qa",
            length_bucket="short",
            ttft_ms=6.0,
            peak_memory_mib=40.0,
            kv_cache_memory_mib=20.0,
            resident_memory_mib=20.0,
            quality_loss=0.01,
            kv_cumulative_mib=20.0,
            resident_kv_mib_before=0.0,
            resident_kv_mib_after=20.0,
            restore_ms=0.0,
            recompute_ms=0.0,
            evicted_kv_mib=0.0,
            budget_hit=False,
        ),
        PolicyRunRecord(
            policy="tailguard",
            request_id="sess-a_turn_001",
            session_id="sess-a",
            turn_index=1,
            action_profile="full_cpu",
            ok=True,
            measured=True,
            task="qa",
            length_bucket="short",
            ttft_ms=8.0,
            peak_memory_mib=80.0,
            kv_cache_memory_mib=40.0,
            resident_memory_mib=10.0,
            quality_loss=0.0,
            kv_cumulative_mib=80.0,
            resident_kv_mib_before=20.0,
            resident_kv_mib_after=10.0,
            restore_ms=5.0,
            recompute_ms=7.0,
            evicted_kv_mib=30.0,
            budget_hit=True,
            fallback_reason="budget",
        ),
    ]

    summary = MetricCollector().summarize_policy_runs(records, epsilon=0.05, delta=0.05, exact_profiles={"full_cpu"})["tailguard"]

    assert summary["switch_count"] == 1.0
    assert summary["budget_hit_rate"] == 0.5
    assert summary["restore_count"] == 1.0
    assert summary["restore_time_ms"] == 5.0
    assert summary["recompute_count"] == 1.0
    assert summary["recompute_time_ms"] == 7.0
    assert summary["mean_resident_kv_mib"] == 15.0
    assert summary["mean_cumulative_kv_mib"] == 50.0
    assert summary["profile_residence_share"] == {"full_cpu": 0.2, "kivi": 0.8}


def test_plot_summary_groups_rows_by_constraint_for_subplots() -> None:
    grouped = plot_summary_module._constraint_groups(
        [
            {"policy": "full_lru", "memory_budget_mib": "4900", "epsilon": "0.05", "delta": "0.05"},
            {"policy": "tailguard", "memory_budget_mib": "5000", "epsilon": "0.05", "delta": "0.05"},
            {"policy": "full_lru", "memory_budget_mib": "4900", "epsilon": "0.10", "delta": "0.05"},
        ]
    )

    assert list(grouped) == [("0.05", "0.05"), ("0.10", "0.05")]
    assert [row["memory_budget_mib"] for row in grouped[("0.05", "0.05")]] == ["4900", "5000"]
