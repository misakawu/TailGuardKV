from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_validate_baseline_quality_fixture_accepts_external_labeled_subset(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_quality.jsonl"
    rows = [
        {
            "request_id": "qa-001",
            "task": "qa",
            "prompt": "Question: q1\nAnswer:",
            "reference": "a1",
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "longbench_qasper",
                "split": "calibration",
                "risk_family": "kivi_sensitive",
            },
        },
        {
            "request_id": "sum-001",
            "task": "summary",
            "prompt": "Summarize doc",
            "reference": "s1",
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "longbench_gov_report",
                "split": "eval",
                "risk_family": "h2o_sensitive",
            },
        },
        {
            "request_id": "code-001",
            "task": "code",
            "prompt": "Write python",
            "reference": "def f(): pass",
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "longbench_lcc",
                "split": "eval",
                "risk_family": "low_risk",
            },
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    from scripts.import_external_fixtures import validate_baseline_quality_fixture

    report = validate_baseline_quality_fixture(fixture_path)

    assert report["row_count"] == 3
    assert set(report["tasks"]) == {"qa", "summary", "code"}
    assert set(report["splits"]) == {"calibration", "eval"}
    assert set(report["risk_families"]) == {"kivi_sensitive", "h2o_sensitive", "low_risk"}


def test_validate_baseline_quality_fixture_rejects_missing_required_metadata(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_quality_bad.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "request_id": "qa-001",
                "task": "qa",
                "prompt": "Question: bad\nAnswer:",
                "reference": "a1",
                "metadata": {
                    "source": "external_labeling",
                    "split": "eval",
                    "risk_family": "kivi_sensitive",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    from scripts.import_external_fixtures import validate_baseline_quality_fixture

    with pytest.raises(ValueError, match="source_dataset"):
        validate_baseline_quality_fixture(fixture_path)


def test_validate_baseline_session_fixture_accepts_session_trace_jsonl(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session.jsonl"
    rows = [
        {
            "request_id": "s1-t0",
            "task": "chat",
            "prompt": "first turn",
            "reference": "r0",
            "session_id": "s1",
            "turn_index": 0,
            "arrival_index": 0,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "calibration",
                "risk_family": "kivi_sensitive",
            },
        },
        {
            "request_id": "s2-t0",
            "task": "chat",
            "prompt": "other session",
            "reference": "r0",
            "session_id": "s2",
            "turn_index": 0,
            "arrival_index": 1,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "eval",
                "risk_family": "h2o_sensitive",
            },
        },
        {
            "request_id": "s1-t1",
            "task": "chat",
            "prompt": "follow up",
            "reference": "r1",
            "session_id": "s1",
            "turn_index": 1,
            "arrival_index": 2,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "calibration",
                "risk_family": "kivi_sensitive",
            },
        },
        {
            "request_id": "s3-t0",
            "task": "chat",
            "prompt": "low risk start",
            "reference": "r0",
            "session_id": "s3",
            "turn_index": 0,
            "arrival_index": 3,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "eval",
                "risk_family": "low_risk",
            },
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    report = validate_baseline_session_fixture(fixture_path)

    assert report["row_count"] == 4
    assert report["session_count"] == 3
    assert report["multi_turn_session_count"] == 1
    assert report["interleaved_session_count"] >= 2
    assert set(report["risk_families"]) == {"kivi_sensitive", "h2o_sensitive", "low_risk"}


def test_validate_baseline_session_fixture_accepts_low_risk_only_jsonl(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_low_risk.jsonl"
    rows = [
        {
            "request_id": "s1-t0",
            "task": "chat",
            "prompt": "first turn",
            "reference": "r0",
            "session_id": "s1",
            "turn_index": 0,
            "arrival_index": 0,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "calibration",
                "risk_family": "low_risk",
                },
            },
        {
            "request_id": "s2-t0",
            "task": "chat",
            "prompt": "other session",
            "reference": "r0",
            "session_id": "s2",
            "turn_index": 0,
            "arrival_index": 1,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "eval",
                "risk_family": "low_risk",
            },
        },
        {
            "request_id": "s1-t1",
            "task": "chat",
            "prompt": "follow up",
            "reference": "r1",
            "session_id": "s1",
            "turn_index": 1,
            "arrival_index": 2,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "calibration",
                "risk_family": "low_risk",
            },
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    report = validate_baseline_session_fixture(fixture_path)

    assert report["row_count"] == 3
    assert set(report["risk_families"]) == {"low_risk"}


def test_validate_baseline_session_fixture_rejects_non_continuous_turn_index(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_bad.jsonl"
    rows = [
        {
            "request_id": "s1-t0",
            "task": "chat",
            "prompt": "first turn",
            "reference": "r0",
            "session_id": "s1",
            "turn_index": 0,
            "arrival_index": 0,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "calibration",
                "risk_family": "kivi_sensitive",
            },
        },
        {
            "request_id": "s1-t2",
            "task": "chat",
            "prompt": "skip turn",
            "reference": "r2",
            "session_id": "s1",
            "turn_index": 2,
            "arrival_index": 1,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "eval",
                "risk_family": "kivi_sensitive",
            },
        },
        {
            "request_id": "s2-t0",
            "task": "chat",
            "prompt": "other session",
            "reference": "r0",
            "session_id": "s2",
            "turn_index": 0,
            "arrival_index": 2,
            "metadata": {
                "source": "external_labeling",
                "source_dataset": "sharegpt_pilot",
                "split": "eval",
                "risk_family": "low_risk",
            },
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="turn_index"):
        validate_baseline_session_fixture(fixture_path)
