from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from run_util.core_types import ProfileMeasurement
from scripts.generate_pilot_session_trace_requests import (
    MAINSTREAM_H2O_PROFILES,
    MAINSTREAM_KIVI_PROFILES,
    build_requests_from_templates,
)
from scripts.validate_trace_quality import validate_trace_quality


def _template(request_id: str, *, task: str, risk_family: str, split: str, followup_kind: str | None = None) -> dict[str, object]:
    turn_index = 1 if followup_kind else 0
    return {
        "request_id": request_id,
        "task": task,
        "prompt": f"{task} prompt {request_id}",
        "reference": f"{task} reference {request_id}",
        "session_id": f"session-{request_id}",
        "turn_index": turn_index,
        "history_turns": ["User: setup", "Assistant: detail"] if turn_index else [],
        "metadata": {
            "split": split,
            "length_bucket": "long" if task == "summary" else "medium",
            "risk_family": risk_family,
            "risk_profiles": (
                [MAINSTREAM_KIVI_PROFILES[0]]
                if risk_family == "kivi_sensitive"
                else [MAINSTREAM_H2O_PROFILES[0]]
                if risk_family == "h2o_sensitive"
                else []
            ),
            "pressure_phase": "quality" if followup_kind else "memory",
            "followup_kind": followup_kind,
        },
    }


def _measurement(
    request_id: str,
    *,
    profile: str,
    task: str,
    split: str,
    quality_loss: float,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        profile=profile,
        adapter=profile.split("_", 1)[0],
        ok=True,
        measured=True,
        output_text="answer",
        quality_loss=quality_loss,
        extra={"task": task, "split": split},
    )


def test_build_requests_from_templates_enforces_session_trace_layout() -> None:
    templates: list[dict[str, object]] = []
    for index in range(6):
        templates.append(_template(f"qa-base-{index}", task="qa", risk_family="kivi_sensitive", split="calibration"))
        templates.append(
            _template(
                f"qa-follow-{index}",
                task="qa",
                risk_family="kivi_sensitive",
                split="eval",
                followup_kind="constraint_recall",
            )
        )
        templates.append(_template(f"sum-base-{index}", task="summary", risk_family="h2o_sensitive", split="calibration"))
        templates.append(
            _template(
                f"sum-follow-{index}",
                task="summary",
                risk_family="h2o_sensitive",
                split="eval",
                followup_kind="detail_recall",
            )
        )
    templates.extend(
        [
            _template("qa-low", task="qa", risk_family="low_risk", split="eval"),
            _template("sum-low", task="summary", risk_family="low_risk", split="eval"),
        ]
    )

    rows = build_requests_from_templates(templates)

    assert {row["task"] for row in rows} == {"qa", "summary"}
    assert len(rows) == len(templates)
    assert all("metadata" in row for row in rows)
    assert all(
        {"split", "length_bucket", "risk_family", "risk_profiles", "pressure_phase", "followup_kind"}
        <= set(row["metadata"].keys())
        for row in rows
    )
    cutoff = int(len(rows) * 0.6)
    assert all(row["metadata"]["pressure_phase"] == "memory" for row in rows[:cutoff])
    assert all(row["metadata"]["pressure_phase"] == "quality" for row in rows[cutoff:])
    assert any(int(row["turn_index"]) > 0 for row in rows)
    assert any(row["metadata"]["followup_kind"] == "constraint_recall" for row in rows)
    assert any(row["metadata"]["followup_kind"] == "detail_recall" for row in rows)


def test_validate_trace_quality_passes_with_h2o_and_kivi_eval_signal() -> None:
    fixture_rows = [
        _template("qa-eval", task="qa", risk_family="kivi_sensitive", split="eval", followup_kind="constraint_recall"),
        _template("sum-eval", task="summary", risk_family="h2o_sensitive", split="eval", followup_kind="detail_recall"),
    ]
    measurements = [
        _measurement("qa-eval", profile=MAINSTREAM_KIVI_PROFILES[0], task="qa", split="eval", quality_loss=0.04),
        _measurement("sum-eval", profile=MAINSTREAM_H2O_PROFILES[0], task="summary", split="eval", quality_loss=0.03),
    ]

    result = validate_trace_quality(measurements, fixture_rows)

    assert result.passed is True
    assert result.covered_tasks == {"qa", "summary"}
    assert set(result.qualifying_profiles) == {MAINSTREAM_KIVI_PROFILES[0], MAINSTREAM_H2O_PROFILES[0]}


def test_validate_trace_quality_fails_for_single_profile_signal() -> None:
    fixture_rows = [
        _template("qa-eval", task="qa", risk_family="kivi_sensitive", split="eval", followup_kind="constraint_recall"),
        _template("sum-eval", task="summary", risk_family="h2o_sensitive", split="eval", followup_kind="detail_recall"),
    ]
    measurements = [
        _measurement("qa-eval", profile=MAINSTREAM_KIVI_PROFILES[0], task="qa", split="eval", quality_loss=0.04),
        _measurement("sum-eval", profile=MAINSTREAM_KIVI_PROFILES[0], task="summary", split="eval", quality_loss=0.03),
    ]

    result = validate_trace_quality(measurements, fixture_rows)

    assert result.passed is False
    assert "至少需要 2 个不同 profile" in result.errors[0]


def test_validate_trace_quality_fails_when_nonzero_loss_only_covers_one_task() -> None:
    fixture_rows = [
        _template("qa-eval-a", task="qa", risk_family="kivi_sensitive", split="eval", followup_kind="constraint_recall"),
        _template("qa-eval-b", task="qa", risk_family="h2o_sensitive", split="eval", followup_kind="constraint_recall"),
        _template("sum-eval", task="summary", risk_family="h2o_sensitive", split="eval", followup_kind="detail_recall"),
    ]
    measurements = [
        _measurement("qa-eval-a", profile=MAINSTREAM_KIVI_PROFILES[0], task="qa", split="eval", quality_loss=0.04),
        _measurement("qa-eval-b", profile=MAINSTREAM_H2O_PROFILES[0], task="qa", split="eval", quality_loss=0.03),
        _measurement("sum-eval", profile=MAINSTREAM_H2O_PROFILES[0], task="summary", split="eval", quality_loss=0.0),
    ]

    result = validate_trace_quality(measurements, fixture_rows)

    assert result.passed is False
    assert any("QA 与 Summary" in message for message in result.errors)


def test_pilot_session_trace_config_and_fixture_match_reconstructed_dataset() -> None:
    config = yaml.safe_load(Path("configs/pilot_session_trace.yaml").read_text(encoding="utf-8"))
    fixture_path = Path("data/fixtures/pilot_session_trace_requests.jsonl")
    rows = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert config["experiment"]["type"] == "baseline_session"
    assert config["data"]["source"] == "fixture"
    assert config["data"]["requests"] == "data/fixtures/pilot_session_trace_requests.jsonl"
    assert config["data"]["quality_mode"] == "session_diagnostic"
    assert config["data"]["max_requests"] >= 200
    assert len(rows) == config["data"]["max_requests"]
    assert {row["task"] for row in rows} == {"qa", "summary"}
    assert {row["metadata"]["split"] for row in rows} == {"calibration", "eval"}
    assert {"kivi_sensitive", "h2o_sensitive", "low_risk"} <= {row["metadata"]["risk_family"] for row in rows}
    assert all(row["metadata"]["pressure_phase"] in {"memory", "quality"} for row in rows)
    assert any(int(row["turn_index"]) > 0 for row in rows)
    cutoff = int(len(rows) * 0.6)
    assert all(row["metadata"]["pressure_phase"] == "memory" for row in rows[:cutoff])
    assert all(row["metadata"]["pressure_phase"] == "quality" for row in rows[cutoff:])


def test_pilot_session_trace_fixture_eval_keeps_mainstream_h2o_and_kivi_risk_samples() -> None:
    fixture_path = Path("data/fixtures/pilot_session_trace_requests.jsonl")
    rows = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_rows = [row for row in rows if row["metadata"]["split"] == "eval"]

    assert any(profile in row["metadata"]["risk_profiles"] for row in eval_rows for profile in MAINSTREAM_H2O_PROFILES)
    assert any(profile in row["metadata"]["risk_profiles"] for row in eval_rows for profile in MAINSTREAM_KIVI_PROFILES)
    assert any(row["task"] == "qa" and row["metadata"]["pressure_phase"] == "quality" for row in eval_rows)
    assert any(row["task"] == "summary" and row["metadata"]["pressure_phase"] == "quality" for row in eval_rows)

