from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from metrics import MetricCollector
from run_util.config_loader import config_experiment_type, config_runtime, load_config
from run_util.core_types import PolicyRunRecord, ProfileMeasurement, Request
from run_util.build_profile_table import _limit_requests_for_experiment
from run_util.data_utils import limit_requests_by_split, load_requests, validate_requests_for_experiment_type
from run_util.io_utils import write_csv
from run_util.run_policies import run_policies
from run_util.session_trace import synthesize_pressure_trace
from run_util.validation import (
    validate_experiment_policy_records,
    validate_profile_measurements,
)


def _measurement(
    request_id: str = "r1",
    *,
    session_id: str | None = None,
    turn_index: int = 0,
    resident_before: float | None = None,
    resident_after: float | None = None,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        session_id=session_id,
        turn_index=turn_index,
        profile="full_gpu",
        adapter="test",
        ok=True,
        measured=True,
        output_text="answer",
        latency_ms=1.0,
        ttft_ms=1.0,
        peak_memory_mib=10.0,
        kv_cache_memory_mib=10.0,
        resident_kv_mib_before=resident_before,
        resident_kv_mib_after=resident_after,
        quality_loss=0.0,
        extra={
            "task": "qa",
            "length_bucket": "short",
            "split": "eval",
            "arrival_index": 0,
            "prompt_text": "prompt",
            "history_turns": "[]",
            "effective_prompt_chars": 6,
        },
    )


def _record(
    request_id: str = "r1",
    *,
    session_id: str | None = None,
    turn_index: int = 0,
    global_resident: float | None = None,
    budget_hit: bool = False,
    evicted: float | None = None,
    restore_ms: float | None = None,
    recompute_ms: float | None = None,
    queue_ms: float | None = None,
) -> PolicyRunRecord:
    return PolicyRunRecord(
        policy="full_lru",
        request_id=request_id,
        session_id=session_id,
        turn_index=turn_index,
        task="qa",
        length_bucket="short",
        action_profile="full_gpu",
        ok=True,
        measured=True,
        quality_loss=0.0,
        resident_kv_mib_before=0.0,
        resident_kv_mib_after=10.0,
        global_resident_kv_mib=global_resident,
        budget_hit=budget_hit,
        backend_budget_hit=budget_hit,
        evicted_kv_mib=evicted,
        restore_ms=restore_ms,
        recompute_ms=recompute_ms,
        queue_delay_ms=queue_ms,
    )


def test_config_experiment_type_is_explicit_and_legacy_quality_mode_compatible() -> None:
    assert config_experiment_type({"experiment": {"type": "baseline_quality"}}) == "baseline_quality"
    assert config_experiment_type({"data": {"quality_mode": "session_diagnostic"}}) == "baseline_session"
    assert config_experiment_type({"data": {"quality_mode": "baseline"}}) == "baseline_quality"


def test_quality_requests_allow_missing_session_identity() -> None:
    validate_requests_for_experiment_type(
        [Request("r1", "qa", "prompt", turn_index=0)],
        "baseline_quality",
    )


def test_load_requests_infers_session_history_from_prompt_and_reference(tmp_path: Path) -> None:
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "session_a_turn_000",
                        "task": "chat",
                        "prompt": "hello",
                        "reference": "hi",
                        "session_id": "session-a",
                        "turn_index": 0,
                        "arrival_index": 0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "request_id": "session_a_turn_001",
                        "task": "chat",
                        "prompt": "how are you",
                        "reference": "fine",
                        "session_id": "session-a",
                        "turn_index": 1,
                        "arrival_index": 1,
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    requests, _ = load_requests({"data": {"requests": str(requests_path)}})

    assert requests[1].history_turns == ("User: hello", "Assistant: hi")
    assert requests[1].effective_prompt == "User: hello\nAssistant: hi\nhow are you"


def test_session_requests_require_reuse_and_positive_turn() -> None:
    with pytest.raises(ValueError, match="session_id"):
        validate_requests_for_experiment_type(
            [Request("r1", "qa", "prompt", turn_index=0)],
            "baseline_session",
        )


def test_session_requests_reject_out_of_order_turns() -> None:
    with pytest.raises(ValueError, match="turn_index"):
        validate_requests_for_experiment_type(
            [
                Request("s1_t0", "chat", "turn0", session_id="s1", turn_index=0, arrival_index=0),
                Request("s1_t2", "chat", "turn2", session_id="s1", turn_index=2, arrival_index=1),
            ],
            "baseline_session",
        )


def test_limit_requests_by_split_preserves_session_arrival_order() -> None:
    requests = [
        Request("s1_t0", "chat", "turn0", session_id="s1", turn_index=0, arrival_index=0, metadata={"split": "calibration"}),
        Request("s2_t0", "chat", "turn0", session_id="s2", turn_index=0, arrival_index=1, metadata={"split": "calibration"}),
        Request("s1_t1", "chat", "turn1", session_id="s1", turn_index=1, arrival_index=2, metadata={"split": "eval"}),
        Request("s2_t1", "chat", "turn1", session_id="s2", turn_index=1, arrival_index=3, metadata={"split": "eval"}),
        Request("s3_t0", "chat", "turn0", session_id="s3", turn_index=0, arrival_index=4, metadata={"split": "calibration"}),
        Request("s3_t1", "chat", "turn1", session_id="s3", turn_index=1, arrival_index=5, metadata={"split": "eval"}),
    ]

    limited = limit_requests_by_split(requests, 4)

    assert [request.arrival_index for request in limited] == sorted(request.arrival_index for request in limited)
    validate_requests_for_experiment_type(limited, "baseline_session")


def test_pilot_session_trace_limit_preserves_pressure_arrival_order() -> None:
    config = load_config(Path("configs/pilot_session_trace.yaml"))
    requests, _ = load_requests(config)
    trace = synthesize_pressure_trace(requests, copies=2, repeat_rounds=1, memory_budget_mib=18.0)

    limited = _limit_requests_for_experiment(trace, int(config_runtime(config)["max_requests"]), "baseline_session")

    assert [request.arrival_index for request in limited] == sorted(request.arrival_index for request in limited)
    validate_requests_for_experiment_type(limited, "baseline_session")


def test_canonical_probe_is_not_truncated_by_max_requests() -> None:
    trace = [
        Request(
            f"session-{session}_turn-{turn}",
            "qa",
            "prompt",
            session_id=f"session-{session}",
            turn_index=turn,
            arrival_index=turn * 6 + session,
            metadata={"canonical_history_mode": "full_gpu_generated_v1"},
        )
        for turn in range(5)
        for session in range(6)
    ]

    limited = _limit_requests_for_experiment(trace, 12, "baseline_session")

    assert len(limited) == 30


def test_session_profile_validation_requires_resident_fields() -> None:
    with pytest.raises(ValueError, match="resident_kv_mib"):
        validate_profile_measurements(
            [_measurement("r1", session_id="s1", turn_index=1)],
            experiment_type="baseline_session",
        )


def test_quality_summary_marks_backend_metrics_not_applicable() -> None:
    summary = MetricCollector().summarize_policy_runs(
        [_record()],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_gpu"},
        experiment_type="baseline_quality",
    )["full_lru"]

    assert summary["backend_events_applicable"] is False
    assert summary["budget_hit_rate"] is None
    assert summary["backend_semantics_status"] == "not_applicable"


def test_session_policy_validation_requires_backend_event_evidence() -> None:
    with pytest.raises(ValueError, match="backend"):
        validate_experiment_policy_records(
            [
                _record("r1", session_id="s1", turn_index=0, global_resident=10.0),
                _record("r2", session_id="s2", turn_index=1, global_resident=11.0),
            ],
            "baseline_session",
        )


def test_session_policy_validation_accepts_nontrivial_backend_trace() -> None:
    validate_experiment_policy_records(
        [
            _record("r1", session_id="s1", turn_index=0, global_resident=10.0),
            _record("r2", session_id="s2", turn_index=0, global_resident=20.0, budget_hit=True, evicted=2.0),
            _record("r3", session_id="s1", turn_index=1, global_resident=15.0, restore_ms=1.0),
        ],
        "baseline_session",
    )


def test_session_policy_validation_accepts_eval_only_records_with_backend_history() -> None:
    validate_experiment_policy_records(
        [
            _record("r2", session_id="s2", turn_index=0, global_resident=20.0, budget_hit=True, evicted=2.0),
            _record("r3", session_id="s1", turn_index=1, global_resident=15.0, restore_ms=1.0),
        ],
        "baseline_session",
    )


def test_run_policies_replays_full_session_history_but_only_outputs_eval_rows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = tmp / "baseline_session.yaml"
        measurements_path = tmp / "measured.csv"
        output_path = tmp / "policy.csv"
        config_path.write_text(
            "\n".join(
                [
                    "experiment:",
                    "  type: baseline_session",
                    "profiles:",
                    "  names:",
                    "    - full_gpu",
                    "policies:",
                    "  names:",
                    "    - full_lru",
                    "pilot:",
                    "  memory_budgets_mib:",
                    "    - 50",
                ]
            ),
            encoding="utf-8",
        )
        rows = [
            ProfileMeasurement(
                request_id="s1_t0",
                session_id="s1",
                turn_index=0,
                profile="full_gpu",
                adapter="test",
                ok=True,
                measured=True,
                output_text="turn0",
                latency_ms=8.0,
                ttft_ms=8.0,
                peak_memory_mib=30.0,
                kv_cache_memory_mib=30.0,
                resident_memory_mib=30.0,
                kv_incremental_mib=30.0,
                kv_cumulative_mib=30.0,
                resident_kv_mib_before=0.0,
                resident_kv_mib_after=30.0,
                quality_loss=0.0,
                extra={
                    "task": "chat",
                    "length_bucket": "short",
                    "split": "calibration",
                    "arrival_index": 0,
                    "prompt_text": "s1 turn0",
                    "history_turns": json.dumps([]),
                    "effective_prompt_chars": 8,
                },
            ),
            ProfileMeasurement(
                request_id="s2_t0",
                session_id="s2",
                turn_index=0,
                profile="full_gpu",
                adapter="test",
                ok=True,
                measured=True,
                output_text="turn0",
                latency_ms=8.0,
                ttft_ms=8.0,
                peak_memory_mib=30.0,
                kv_cache_memory_mib=30.0,
                resident_memory_mib=30.0,
                kv_incremental_mib=30.0,
                kv_cumulative_mib=30.0,
                resident_kv_mib_before=0.0,
                resident_kv_mib_after=30.0,
                quality_loss=0.0,
                extra={
                    "task": "chat",
                    "length_bucket": "short",
                    "split": "calibration",
                    "arrival_index": 1,
                    "prompt_text": "s2 turn0",
                    "history_turns": json.dumps([]),
                    "effective_prompt_chars": 8,
                },
            ),
            ProfileMeasurement(
                request_id="s1_t1",
                session_id="s1",
                turn_index=1,
                profile="full_gpu",
                adapter="test",
                ok=True,
                measured=True,
                output_text="turn1",
                latency_ms=10.0,
                ttft_ms=10.0,
                peak_memory_mib=35.0,
                kv_cache_memory_mib=20.0,
                resident_memory_mib=20.0,
                kv_incremental_mib=5.0,
                kv_cumulative_mib=35.0,
                resident_kv_mib_before=30.0,
                resident_kv_mib_after=20.0,
                quality_loss=0.0,
                extra={
                    "task": "chat",
                    "length_bucket": "short",
                    "split": "eval",
                    "arrival_index": 2,
                    "prompt_text": "s1 turn1",
                    "history_turns": json.dumps(["User: s1 turn0", "Assistant: reply0"]),
                    "effective_prompt_chars": 32,
                },
            ),
            ProfileMeasurement(
                request_id="s2_t1",
                session_id="s2",
                turn_index=1,
                profile="full_gpu",
                adapter="test",
                ok=True,
                measured=True,
                output_text="turn1",
                latency_ms=10.0,
                ttft_ms=10.0,
                peak_memory_mib=35.0,
                kv_cache_memory_mib=20.0,
                resident_memory_mib=20.0,
                kv_incremental_mib=5.0,
                kv_cumulative_mib=35.0,
                resident_kv_mib_before=30.0,
                resident_kv_mib_after=20.0,
                quality_loss=0.0,
                extra={
                    "task": "chat",
                    "length_bucket": "short",
                    "split": "eval",
                    "arrival_index": 3,
                    "prompt_text": "s2 turn1",
                    "history_turns": json.dumps(["User: s2 turn0", "Assistant: reply0"]),
                    "effective_prompt_chars": 32,
                },
            ),
        ]
        write_csv(measurements_path, [row.to_row() for row in rows])
        args = argparse.Namespace(
            config=str(config_path),
            measurements=str(measurements_path),
            output=str(output_path),
            profiles=None,
            policies=None,
            epsilon=0.2,
            delta=0.05,
            memory_budget_mib=50.0,
            allow_dry_run_replay=False,
            use_pandas_replay=False,
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = run_policies(args)

        assert code == 0, stream.getvalue()
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        assert [record["request_id"] for record in records] == ["s1_t1", "s2_t1"]
        assert all(record["session_id"] in {"s1", "s2"} for record in records)
        assert any(
            float(record["restore_ms"] or 0.0) > 0.0
            or (record["budget_hit"] == "True")
            or float(record["evicted_kv_mib"] or 0.0) > 0.0
            or float(record["queue_delay_ms"] or 0.0) > 0.0
            for record in records
        )
