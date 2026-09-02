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


def _hybrid_session_rows(*, content_source_dataset: str = "longbench") -> list[dict[str, object]]:
    sessions: list[list[dict[str, object]]] = []
    session_index = 0
    roles = ("sharegpt_opening", "sharegpt_opening", "content_query", "reference_recall", "reference_rewrite")
    for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk"):
        for task in ("qa", "summary"):
            for split in ("calibration", "eval"):
                for _ in range(4):
                    session_id = f"hybrid-session-{session_index:03d}"
                    content_request_id = f"longbench-{session_index:03d}"
                    reference = f"reference-{session_index:03d}"
                    session_rows: list[dict[str, object]] = []
                    for turn_index in range(5):
                        injected = turn_index >= 2
                        session_rows.append(
                            {
                                "request_id": f"{session_id}-turn-{turn_index}",
                                "task": task if injected else "chat",
                                "prompt": f"prompt {session_index} {turn_index}",
                                "reference": reference if injected else f"opening {session_index} {turn_index}",
                                "session_id": session_id,
                                "turn_index": turn_index,
                                "metadata": {
                                    "source": "hybrid_session_builder",
                                    "source_dataset": f"sharegpt_{content_source_dataset}_hybrid_session",
                                    "content_source_dataset": content_source_dataset if injected else "sharegpt",
                                    "content_source_request_id": content_request_id if injected else f"sharegpt-{session_index}-{turn_index}",
                                    "content_source_index": session_index if injected else session_index * 2 + turn_index,
                                    "content_payload_hash": f"payload-{session_index:03d}" if injected else "",
                                    "injection_template": "template_a",
                                    "original_session_id": f"sharegpt-session-{session_index:03d}",
                                    "hybrid_turn_role": roles[turn_index],
                                    "split": split,
                                    "risk_family": risk_family,
                                },
                            }
                        )
                        if injected and content_source_dataset == "raghot_qa":
                            session_rows[-1]["metadata"].update(
                                {
                                    "context_pack_hash": f"context-{session_index:03d}",
                                    "supporting_fact_ids": [f"fact-{session_index:03d}"],
                                    "packing_policy_version": "raghot_support_first_v1",
                                }
                            )
                    sessions.append(session_rows)
                    session_index += 1

    rows: list[dict[str, object]] = []
    for turn_index in range(5):
        for session_rows in sessions:
            row = dict(session_rows[turn_index])
            row["arrival_index"] = len(rows)
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_validate_baseline_session_fixture_accepts_exact_hybrid_contract(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session.jsonl"
    _write_rows(fixture_path, _hybrid_session_rows())

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    report = validate_baseline_session_fixture(fixture_path)

    assert report["row_count"] == 240
    assert report["session_count"] == 48
    assert report["turns_per_session"] == 5
    assert set(report["risk_families"]) == {"kivi_sensitive", "h2o_sensitive", "low_risk"}
    assert set(report["risk_task_split_session_counts"].values()) == {4}


def test_validate_baseline_session_fixture_accepts_raghot_qa_provenance(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_raghot.jsonl"
    _write_rows(fixture_path, _hybrid_session_rows(content_source_dataset="raghot_qa"))

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    assert validate_baseline_session_fixture(fixture_path)["row_count"] == 240


def test_validate_baseline_session_fixture_rejects_missing_hybrid_metadata(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_missing_metadata.jsonl"
    rows = _hybrid_session_rows()
    del rows[0]["metadata"]["original_session_id"]
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="original_session_id"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_missing_longbench_provenance(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_bad_longbench.jsonl"
    rows = _hybrid_session_rows()
    injected_row = next(row for row in rows if row["turn_index"] == 2)
    injected_row["metadata"]["content_source_dataset"] = "sharegpt"
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="content source"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_missing_raghot_evidence(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_missing_raghot_evidence.jsonl"
    rows = _hybrid_session_rows(content_source_dataset="raghot_qa")
    del next(row for row in rows if row["turn_index"] == 2)["metadata"]["context_pack_hash"]
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="context_pack_hash"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_forged_source_and_role(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_forged_source.jsonl"
    rows = _hybrid_session_rows()
    next(row for row in rows if row["turn_index"] == 2)["metadata"]["content_source_dataset"] = "forged_source"
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="content source"):
        validate_baseline_session_fixture(fixture_path)

    next(row for row in rows if row["turn_index"] == 2)["metadata"]["content_source_dataset"] = "longbench"
    next(row for row in rows if row["turn_index"] == 2)["metadata"]["hybrid_turn_role"] = "reference_recall"
    _write_rows(fixture_path, rows)
    with pytest.raises(ValueError, match="hybrid_turn_role"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_chat_as_injected_risk_sample(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_chat_risk.jsonl"
    rows = _hybrid_session_rows()
    injected_row = next(row for row in rows if row["turn_index"] == 2)
    injected_row["task"] = "chat"
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="QA/Summary"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_wrong_session_split_count(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_bad_split_count.jsonl"
    rows = _hybrid_session_rows()
    session_id = str(rows[0]["session_id"])
    for row in rows:
        if row["session_id"] == session_id:
            row["metadata"]["split"] = "eval"
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="risk/task/split"):
        validate_baseline_session_fixture(fixture_path)


@pytest.mark.parametrize("request_id", ["", "   "])
def test_validate_baseline_session_fixture_rejects_blank_request_id(tmp_path: Path, request_id: str) -> None:
    fixture_path = tmp_path / "baseline_session_blank_request_id.jsonl"
    rows = _hybrid_session_rows()
    rows[0]["request_id"] = request_id
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="request_id.*不能为空"):
        validate_baseline_session_fixture(fixture_path)


def test_validate_baseline_session_fixture_rejects_duplicate_request_id(tmp_path: Path) -> None:
    fixture_path = tmp_path / "baseline_session_duplicate_request_id.jsonl"
    rows = _hybrid_session_rows()
    rows[1]["request_id"] = rows[0]["request_id"]
    _write_rows(fixture_path, rows)

    from scripts.import_external_fixtures import validate_baseline_session_fixture

    with pytest.raises(ValueError, match="request_id.*不能重复"):
        validate_baseline_session_fixture(fixture_path)
