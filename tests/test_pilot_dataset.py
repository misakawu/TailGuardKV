from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from metrics import MetricCollector
from run_util.core_types import PolicyRunRecord, Request
from run_util.data_utils import limit_requests_by_split, load_requests


def test_load_requests_normalizes_nested_metadata_schema(tmp_path: Path) -> None:
    request_path = tmp_path / "pilot.jsonl"
    request_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "qa-001",
                        "task": "qa",
                        "prompt": "Question: test?\nAnswer:",
                        "reference": "answer",
                        "metadata": {
                            "source_dataset": "longbench_qasper",
                            "split": "calibration",
                            "length_bucket": "short",
                            "arrival_index": 7,
                            "rag_chunks": [{"chunk_id": "c1", "source": "doc-1"}],
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "request_id": "sum-001",
                        "task": "summary",
                        "prompt": "Summarize:\ntext\nSummary:",
                        "reference": "summary",
                        "metadata": {
                            "source_dataset": "xsum",
                            "split": "eval",
                            "length_bucket": "nonshort",
                            "arrival_index": 8,
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    requests, fallback = load_requests({"data": {"requests": str(request_path)}})

    assert fallback is False
    assert [request.task for request in requests] == ["qa", "summary"]
    assert requests[0].arrival_index == 7
    assert requests[0].metadata["source_dataset"] == "longbench_qasper"
    assert requests[0].metadata["split"] == "calibration"
    assert requests[0].metadata["rag_chunks"] == [{"chunk_id": "c1", "source": "doc-1"}]
    assert requests[1].arrival_index == 8
    assert requests[1].metadata["length_bucket"] == "nonshort"


def test_load_requests_baseline_rejects_missing_reference_when_exact_profile_active(tmp_path: Path) -> None:
    request_path = tmp_path / "pilot.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "request_id": "qa-001",
                "task": "qa",
                "prompt": "Question: test?\nAnswer:",
                "metadata": {
                    "source_dataset": "longbench_qasper",
                    "split": "eval",
                    "length_bucket": "short",
                    "arrival_index": 1,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少 reference"):
        requests, _ = load_requests({"data": {"requests": str(request_path)}})
        from run_util.data_utils import validate_requests_for_quality_mode

        validate_requests_for_quality_mode(requests, "baseline", {"full_gpu"})


def test_limit_requests_by_split_preserves_both_tasks_when_truncated() -> None:
    requests: list[Request] = []
    for split in ("calibration", "eval"):
        for task in ("qa", "summary"):
            for bucket in ("short", "nonshort"):
                for index in range(4):
                    requests.append(
                        Request(
                            request_id=f"{split}-{task}-{bucket}-{index}",
                            task=task,
                            prompt=f"{task} prompt {index}",
                            reference=f"{task} ref {index}",
                            metadata={
                                "split": split,
                                "length_bucket": bucket,
                                "arrival_index": len(requests),
                                "source_dataset": f"{task}_dataset",
                            },
                        )
                    )

    limited = limit_requests_by_split(requests, 8)

    assert len(limited) == 8
    for split in ("calibration", "eval"):
        split_rows = [request for request in limited if request.metadata["split"] == split]
        assert len(split_rows) == 4
        assert {request.task for request in split_rows} == {"qa", "summary"}
        assert {request.metadata["length_bucket"] for request in split_rows} == {"short", "nonshort"}


def test_metric_collector_reports_task_split_and_length_distributions() -> None:
    records = [
        PolicyRunRecord(
            policy="tailguard",
            request_id="qa-short-1",
            session_id="s1",
            turn_index=0,
            task="qa",
            length_bucket="short",
            action_profile="full_gpu",
            ok=True,
            measured=True,
            quality_loss=0.0,
        ),
        PolicyRunRecord(
            policy="tailguard",
            request_id="sum-nonshort-1",
            session_id="s2",
            turn_index=0,
            task="summary",
            length_bucket="nonshort",
            action_profile="kivi_4bit_residual32",
            ok=True,
            measured=True,
            quality_loss=0.2,
        ),
    ]

    summary = MetricCollector().summarize_policy_runs(records, epsilon=0.05, delta=0.05, exact_profiles={"full_gpu"})["tailguard"]

    assert summary["task_distribution"] == {"qa": 1, "summary": 1}
    assert summary["length_bucket_distribution"] == {"nonshort": 1, "short": 1}
    assert summary["task_length_distribution"] == {"qa/short": 1, "summary/nonshort": 1}


def test_pilot_config_points_to_qa_summary_fixture() -> None:
    config = yaml.safe_load(Path("configs/pilot.yaml").read_text(encoding="utf-8"))

    assert config["data"]["requests"] == "data/fixtures/pilot_qa_summary_requests.jsonl"
    assert config["data"]["quality_mode"] == "baseline"
    assert config["data"]["max_requests"] == 200


def test_repo_pilot_fixture_covers_required_task_split_and_length_groups() -> None:
    path = Path("data/fixtures/pilot_qa_summary_requests.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) >= 400
    assert {row["task"] for row in rows} == {"qa", "summary"}
    assert {
        (
            row["metadata"]["split"],
            row["task"],
        )
        for row in rows
    } >= {
        ("calibration", "qa"),
        ("calibration", "summary"),
        ("eval", "qa"),
        ("eval", "summary"),
    }

    eval_rows = [row for row in rows if row["metadata"]["split"] == "eval"]
    assert len(eval_rows) >= 80
    assert len([row for row in eval_rows if row["task"] == "qa"]) >= 40
    assert len([row for row in eval_rows if row["task"] == "summary"]) >= 40

    groups = {
        (row["task"], row["metadata"]["length_bucket"])
        for row in rows
    }
    assert groups >= {
        ("qa", "short"),
        ("qa", "nonshort"),
        ("summary", "short"),
        ("summary", "nonshort"),
    }
