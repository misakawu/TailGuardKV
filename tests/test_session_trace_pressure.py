from __future__ import annotations

from pathlib import Path
import argparse
import csv
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from run_util.core_types import ProfileMeasurement, Request
from run_util.experiment import pilot_smoke_measured
from run_util.io_utils import write_csv
from run_util.run_policies import run_policies
from run_util.session_trace import synthesize_pressure_trace


def _request(
    request_id: str,
    *,
    session_id: str,
    turn_index: int,
    arrival_index: int,
    split: str,
    prompt: str | None = None,
    history_turns: tuple[str, ...] = (),
) -> Request:
    return Request(
        request_id=request_id,
        session_id=session_id,
        task="chat",
        prompt=prompt or request_id,
        turn_index=turn_index,
        arrival_index=arrival_index,
        history_turns=history_turns,
        metadata={
            "split": split,
            "task": "chat",
            "length_bucket": "short",
        },
    )


def _seed_requests() -> list[Request]:
    return [
        _request("a0", session_id="a", turn_index=0, arrival_index=0, split="calibration"),
        _request("b0", session_id="b", turn_index=0, arrival_index=1, split="calibration"),
        _request("a1", session_id="a", turn_index=1, arrival_index=2, split="eval", history_turns=("User: a0", "Assistant: ra0")),
        _request("b1", session_id="b", turn_index=1, arrival_index=3, split="eval", history_turns=("User: b0", "Assistant: rb0")),
    ]


def _trace_measurements(trace: list[Request]) -> list[ProfileMeasurement]:
    measurements: list[ProfileMeasurement] = []
    for request in trace:
        turn_resident = 30.0
        if request.turn_index > 0:
            original_session = str(request.metadata.get("original_session_id") or request.session_id or "")
            turn_resident = 45.0 if original_session.startswith("a") else 35.0
        measurements.append(
            ProfileMeasurement(
                request_id=request.request_id,
                session_id=request.session_id,
                turn_index=request.turn_index,
                profile="full_gpu",
                adapter="test",
                ok=True,
                measured=True,
                output_text="answer",
                latency_ms=10.0,
                ttft_ms=10.0,
                peak_memory_mib=turn_resident,
                kv_cache_memory_mib=turn_resident,
                resident_memory_mib=turn_resident,
                kv_incremental_mib=turn_resident if request.turn_index == 0 else turn_resident - 30.0,
                kv_cumulative_mib=turn_resident,
                resident_kv_mib_before=0.0 if request.turn_index == 0 else 30.0,
                resident_kv_mib_after=turn_resident,
                quality_loss=0.0,
                extra={
                    "task": request.task,
                    "length_bucket": request.metadata.get("length_bucket", "short"),
                    "split": request.metadata.get("split", "eval"),
                    "arrival_index": request.arrival_index,
                    "prompt_text": request.prompt,
                    "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                    "effective_prompt_chars": request.prompt_chars,
                },
            )
        )
    return measurements


def test_pressure_trace_expands_request_flow_and_preserves_history() -> None:
    trace = synthesize_pressure_trace(
        _seed_requests(),
        copies=2,
        repeat_rounds=2,
        memory_budget_mib=50.0,
    )

    assert [request.arrival_index for request in trace] == sorted(request.arrival_index for request in trace)
    assert len({request.session_id for request in trace}) == 8
    assert all("__pressure" in (request.session_id or "") for request in trace)
    copied_turn = next(request for request in trace if request.turn_index == 1)
    assert copied_turn.history_turns
    assert copied_turn.metadata["pressure_copy"] in {1, 2}


def test_pressure_trace_rejects_non_session_seed() -> None:
    seed = [_request("r0", session_id="a", turn_index=0, arrival_index=0, split="eval")]

    with pytest.raises(ValueError, match="至少两个"):
        synthesize_pressure_trace(seed, copies=1, repeat_rounds=1, memory_budget_mib=50.0)


def test_pressure_trace_does_not_change_seed_rows() -> None:
    seed = _seed_requests()
    original = [(row.request_id, row.session_id, row.arrival_index, row.history_turns, dict(row.metadata)) for row in seed]

    synthesize_pressure_trace(seed, copies=2, repeat_rounds=1, memory_budget_mib=50.0)

    assert [(row.request_id, row.session_id, row.arrival_index, row.history_turns, dict(row.metadata)) for row in seed] == original


def test_baseline_session_runner_sends_pressure_trace_to_policy_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    profile_path = tmp_path / "profiles.csv"
    session_trace_path = tmp_path / "session_trace.csv"
    policy_path = tmp_path / "policy.csv"
    summary_path = tmp_path / "summary.csv"
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "\n".join(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "task": "qa" if request.request_id.startswith("a") else "summary",
                        "prompt": request.prompt,
                        "session_id": request.session_id,
                        "turn_index": request.turn_index,
                        "arrival_index": request.arrival_index,
                        "reference": f"reply-{request.request_id}",
                        "history_turns": list(request.history_turns),
                        "metadata": {
                            "split": request.metadata.get("split", "eval"),
                            "length_bucket": "medium" if request.request_id.startswith("a") else "long",
                            "risk_family": "kivi_sensitive" if request.request_id.startswith("a") else "h2o_sensitive",
                            "risk_profiles": ["kivi_2bit_residual32"] if request.request_id.startswith("a") else ["h2o_heavy15_recent15"],
                            "pressure_phase": "quality" if request.turn_index > 0 else "memory",
                            "followup_kind": "constraint_recall" if request.request_id.startswith("a") and request.turn_index > 0 else "detail_recall" if request.turn_index > 0 else "",
                        },
                    },
                    ensure_ascii=False,
                )
                for request in _seed_requests()
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  type: baseline_session",
                "data:",
                f"  requests: {requests_path}",
                "profiles:",
                "  adapters: [test]",
                "  names: [full_gpu, kivi_2bit_residual32, h2o_heavy15_recent15]",
                "policies:",
                "  names: [full_lru]",
                "pilot:",
                "  memory_budgets_mib: [50]",
                "outputs:",
                f"  smoke_profiles: {profile_path}",
                f"  smoke_session_trace: {session_trace_path}",
                f"  smoke_policy: {policy_path}",
                f"  smoke_summary: {summary_path}",
                f"  smoke_split_validation: {tmp_path / 'split_validation.html'}",
            ]
                ),
            encoding="utf-8",
        )
    seed = _seed_requests()
    captured: list[str] = []

    def fake_profile(args: argparse.Namespace) -> int:
        assert len(args.preloaded_requests) == len(seed) * 2
        assert any("__pressure" in (request.session_id or "") for request in args.preloaded_requests)
        rows: list[dict[str, object]] = []
        for request in args.preloaded_requests:
            task = "qa" if request.request_id.startswith("a") else "summary"
            split = request.metadata.get("split", "eval")
            for profile, quality_loss, peak_memory in (
                ("full_gpu", 0.0, 30.0),
                ("kivi_2bit_residual32", 0.05 if task == "qa" else 0.0, 20.0),
                ("h2o_heavy15_recent15", 0.05 if task == "summary" else 0.0, 18.0),
            ):
                rows.append(
                    ProfileMeasurement(
                        request_id=request.request_id,
                        session_id=request.session_id,
                        turn_index=request.turn_index,
                        profile=profile,
                        adapter="test",
                        ok=True,
                        measured=True,
                        output_text="answer",
                        latency_ms=10.0,
                        ttft_ms=10.0,
                        peak_memory_mib=peak_memory,
                        kv_cache_memory_mib=peak_memory,
                        resident_memory_mib=peak_memory,
                        kv_incremental_mib=peak_memory,
                        kv_cumulative_mib=peak_memory,
                        resident_kv_mib_before=10.0 if request.turn_index > 0 else 0.0,
                        resident_kv_mib_after=peak_memory,
                        quality_loss=quality_loss,
                        extra={
                            "task": task,
                            "length_bucket": "medium",
                            "split": split,
                            "arrival_index": request.arrival_index,
                            "prompt_text": request.prompt,
                            "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                            "effective_prompt_chars": request.prompt_chars,
                            "original_request_id": request.metadata.get("original_request_id", request.request_id),
                        },
                    ).to_row()
                )
        write_csv(Path(args.output), rows)
        print(json.dumps({"output": args.output, "rows": len(rows)}))
        return 0

    def fake_policy(args: argparse.Namespace) -> int:
        captured.append(args.measurements)
        print(json.dumps({"output": args.output, "rows": 2, "epsilon": 0.2, "delta": 0.05, "memory_budget_mib": 50}))
        return 0

    with patch("run_util.experiment.build_profile_table", side_effect=fake_profile), patch(
        "run_util.experiment.run_policies", side_effect=fake_policy
    ), patch("run_util.experiment.plot_summary", return_value=[]):
        code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=None, total_summary_output=""))

    assert code == 0
    assert len(captured) == 1
    assert captured[0] == str(profile_path)
    trace_rows = list(__import__("csv").DictReader(session_trace_path.open(encoding="utf-8")))
    assert len(trace_rows) == len(seed) * 2


def test_baseline_session_pressure_trace_produces_backend_event_evidence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    measurements_path = tmp_path / "session_trace.csv"
    output_path = tmp_path / "policy.csv"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  type: baseline_session",
                "profiles:",
                "  names: [full_gpu]",
                "policies:",
                "  names: [full_lru]",
                "pilot:",
                "  memory_budgets_mib: [50]",
            ]
        ),
        encoding="utf-8",
    )
    trace = synthesize_pressure_trace(
        _seed_requests(),
        copies=2,
        repeat_rounds=1,
        memory_budget_mib=50.0,
    )
    write_csv(measurements_path, [row.to_row() for row in _trace_measurements(trace)])

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = run_policies(
            argparse.Namespace(
                config=str(config_path),
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=None,
                policies=None,
                policy_config=None,
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=50.0,
                use_pandas_replay=False,
                allow_dry_run_replay=False,
            )
        )

    assert code == 0, stream.getvalue()
    records = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert records
    assert any(
        record["budget_hit"] == "True"
        or float(record["evicted_kv_mib"] or 0.0) > 0.0
        or float(record["restore_ms"] or 0.0) > 0.0
        or float(record["recompute_ms"] or 0.0) > 0.0
        or float(record["queue_delay_ms"] or 0.0) > 0.0
        for record in records
    )


def test_baseline_session_quality_gate_blocks_policy_sweep_when_eval_signal_is_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    profile_path = tmp_path / "profiles.csv"
    session_trace_path = tmp_path / "session_trace.csv"
    policy_path = tmp_path / "policy.csv"
    summary_path = tmp_path / "summary.csv"
    gate_path = tmp_path / "trace_quality_gate.json"
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task": "qa" if request.request_id.startswith("a") else "summary",
                    "prompt": request.prompt,
                    "session_id": request.session_id,
                    "turn_index": request.turn_index,
                    "arrival_index": request.arrival_index,
                    "reference": f"reply-{request.request_id}",
                    "history_turns": list(request.history_turns),
                    "metadata": {
                        "split": request.metadata.get("split", "eval"),
                        "length_bucket": "medium",
                        "risk_family": "kivi_sensitive" if request.request_id.startswith("a") else "h2o_sensitive",
                        "risk_profiles": ["kivi_2bit_residual32"] if request.request_id.startswith("a") else ["h2o_heavy15_recent15"],
                        "pressure_phase": "quality" if request.turn_index > 0 else "memory",
                        "followup_kind": "constraint_recall" if request.request_id.startswith("a") and request.turn_index > 0 else "detail_recall" if request.turn_index > 0 else "",
                    },
                },
                ensure_ascii=False,
            )
            for request in _seed_requests()
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  type: baseline_session",
                "data:",
                "  source: fixture",
                "  quality_mode: session_diagnostic",
                f"  requests: {requests_path}",
                "  max_requests: 4",
                "profiles:",
                "  adapters: [test]",
                "  names: [full_gpu, kivi_2bit_residual32, h2o_heavy15_recent15]",
                "policies:",
                "  names: [full_lru]",
                "pilot:",
                "  memory_budgets_mib: [50]",
                "outputs:",
                f"  smoke_profiles: {profile_path}",
                f"  smoke_session_trace: {session_trace_path}",
                f"  smoke_policy: {policy_path}",
                f"  smoke_summary: {summary_path}",
                f"  smoke_trace_quality_gate: {gate_path}",
                f"  smoke_split_validation: {tmp_path / 'split_validation.html'}",
            ]
        ),
        encoding="utf-8",
    )

    def fake_profile(args: argparse.Namespace) -> int:
        rows = []
        for request in args.preloaded_requests:
            task = "qa" if request.request_id.startswith("a") else "summary"
            split = request.metadata.get("split", "eval")
            rows.append(
                ProfileMeasurement(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    turn_index=request.turn_index,
                    profile="full_gpu",
                    adapter="test",
                    ok=True,
                    measured=True,
                    output_text="answer",
                    latency_ms=10.0,
                    ttft_ms=10.0,
                    peak_memory_mib=30.0,
                    kv_cache_memory_mib=30.0,
                    resident_memory_mib=30.0,
                    kv_incremental_mib=10.0,
                    kv_cumulative_mib=30.0,
                    resident_kv_mib_before=10.0 if request.turn_index > 0 else 0.0,
                    resident_kv_mib_after=30.0,
                    quality_loss=0.0,
                    extra={
                        "task": task,
                        "length_bucket": "medium",
                        "split": split,
                        "arrival_index": request.arrival_index,
                        "prompt_text": request.prompt,
                        "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                        "effective_prompt_chars": request.prompt_chars,
                        "original_request_id": request.metadata.get("original_request_id", request.request_id),
                    },
                ).to_row()
            )
            rows.append(
                ProfileMeasurement(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    turn_index=request.turn_index,
                    profile="kivi_2bit_residual32",
                    adapter="test",
                    ok=True,
                    measured=True,
                    output_text="answer",
                    latency_ms=8.0,
                    ttft_ms=8.0,
                    peak_memory_mib=20.0,
                    kv_cache_memory_mib=20.0,
                    resident_memory_mib=20.0,
                    kv_incremental_mib=8.0,
                    kv_cumulative_mib=20.0,
                    resident_kv_mib_before=8.0 if request.turn_index > 0 else 0.0,
                    resident_kv_mib_after=20.0,
                    quality_loss=0.05 if task == "qa" else 0.0,
                    extra={
                        "task": task,
                        "length_bucket": "medium",
                        "split": split,
                        "arrival_index": request.arrival_index,
                        "prompt_text": request.prompt,
                        "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                        "effective_prompt_chars": request.prompt_chars,
                        "original_request_id": request.metadata.get("original_request_id", request.request_id),
                    },
                ).to_row()
            )
            rows.append(
                ProfileMeasurement(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    turn_index=request.turn_index,
                    profile="h2o_heavy15_recent15",
                    adapter="test",
                    ok=True,
                    measured=True,
                    output_text="answer",
                    latency_ms=8.5,
                    ttft_ms=8.5,
                    peak_memory_mib=18.0,
                    kv_cache_memory_mib=18.0,
                    resident_memory_mib=18.0,
                    kv_incremental_mib=7.0,
                    kv_cumulative_mib=18.0,
                    resident_kv_mib_before=7.0 if request.turn_index > 0 else 0.0,
                    resident_kv_mib_after=18.0,
                    quality_loss=0.0,
                    extra={
                        "task": task,
                        "length_bucket": "medium",
                        "split": split,
                        "arrival_index": request.arrival_index,
                        "prompt_text": request.prompt,
                        "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                        "effective_prompt_chars": request.prompt_chars,
                        "original_request_id": request.metadata.get("original_request_id", request.request_id),
                    },
                    ).to_row()
                )
        write_csv(Path(args.output), rows)
        print(json.dumps({"output": args.output, "rows": len(rows)}))
        return 0

    with patch("run_util.experiment.build_profile_table", side_effect=fake_profile), patch(
        "run_util.experiment.run_policies", side_effect=AssertionError("policy sweep should be blocked by quality gate")
    ), patch("run_util.experiment.plot_summary", return_value=[]):
        code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=None, total_summary_output=""))

    assert code == 2
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate_payload["passed"] is False
    assert any("H2O" in message for message in gate_payload["errors"])


def test_baseline_session_split_gate_blocks_policy_sweep_when_ks_is_too_high(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    profile_path = tmp_path / "profiles.csv"
    session_trace_path = tmp_path / "session_trace.csv"
    policy_path = tmp_path / "policy.csv"
    summary_path = tmp_path / "summary.csv"
    split_gate_path = tmp_path / "split_validation.html"
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "task": "qa" if request.request_id.startswith("a") else "summary",
                    "prompt": request.prompt,
                    "session_id": request.session_id,
                    "turn_index": request.turn_index,
                    "arrival_index": request.arrival_index,
                    "reference": f"reply-{request.request_id}",
                    "history_turns": list(request.history_turns),
                    "metadata": {
                        "split": request.metadata.get("split", "eval"),
                        "length_bucket": "medium",
                        "risk_family": "kivi_sensitive" if request.request_id.startswith("a") else "h2o_sensitive",
                        "risk_profiles": ["kivi_2bit_residual32"] if request.request_id.startswith("a") else ["h2o_heavy15_recent15"],
                        "pressure_phase": "quality" if request.turn_index > 0 else "memory",
                        "followup_kind": "constraint_recall" if request.request_id.startswith("a") and request.turn_index > 0 else "detail_recall" if request.turn_index > 0 else "",
                    },
                },
                ensure_ascii=False,
            )
            for request in _seed_requests()
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  type: baseline_session",
                "data:",
                "  source: fixture",
                "  quality_mode: session_diagnostic",
                f"  requests: {requests_path}",
                "  max_requests: 4",
                "profiles:",
                "  adapters: [test]",
                "  names: [full_gpu, kivi_2bit_residual32, h2o_heavy15_recent15]",
                "policies:",
                "  names: [full_lru]",
                "pilot:",
                "  memory_budgets_mib: [50]",
                "outputs:",
                f"  smoke_profiles: {profile_path}",
                f"  smoke_session_trace: {session_trace_path}",
                f"  smoke_policy: {policy_path}",
                f"  smoke_summary: {summary_path}",
                f"  smoke_split_validation: {split_gate_path}",
            ]
        ),
        encoding="utf-8",
    )

    def fake_profile(args: argparse.Namespace) -> int:
        rows = []
        for request in args.preloaded_requests:
            task = "qa" if request.request_id.startswith("a") else "summary"
            split = request.metadata.get("split", "eval")
            for profile, quality_loss, peak_memory in (
                ("full_gpu", 0.0, 30.0),
                ("kivi_2bit_residual32", 0.0, 20.0),
                ("h2o_heavy15_recent15", 0.0, 18.0),
            ):
                rows.append(
                    ProfileMeasurement(
                        request_id=request.request_id,
                        session_id=request.session_id,
                        turn_index=request.turn_index,
                        profile=profile,
                        adapter="test",
                        ok=True,
                        measured=True,
                        output_text="answer",
                        latency_ms=10.0,
                        ttft_ms=10.0,
                        peak_memory_mib=peak_memory,
                        kv_cache_memory_mib=peak_memory,
                        resident_memory_mib=peak_memory,
                        kv_incremental_mib=10.0,
                        kv_cumulative_mib=peak_memory,
                        resident_kv_mib_before=10.0 if request.turn_index > 0 else 0.0,
                        resident_kv_mib_after=peak_memory,
                        quality_loss=quality_loss,
                        extra={
                            "task": task,
                            "length_bucket": "medium",
                            "split": split,
                            "arrival_index": request.arrival_index,
                            "prompt_text": request.prompt,
                            "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                            "effective_prompt_chars": request.prompt_chars,
                            "original_request_id": request.metadata.get("original_request_id", request.request_id),
                        },
                    ).to_row()
                )
        write_csv(Path(args.output), rows)
        print(json.dumps({"output": args.output, "rows": len(rows)}))
        return 0

    class ForcedSplitResult:
        passed = False
        errors = ["forced ks failure"]
        html = "<html>forced</html>"

        def to_json(self) -> dict[str, object]:
            return {"passed": False, "errors": ["forced ks failure"], "html": "<html>forced</html>"}

    with patch("run_util.experiment.build_profile_table", side_effect=fake_profile), patch(
        "run_util.experiment.validate_split_balance", return_value=ForcedSplitResult()
    ), patch("run_util.experiment.run_policies") as run_policies_mock, patch(
        "run_util.experiment.plot_summary", return_value=[]
    ):
        code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=None, total_summary_output=""))

    assert code == 2
    run_policies_mock.assert_not_called()
    assert split_gate_path.exists()
