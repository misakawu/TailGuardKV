from __future__ import annotations

import json
from pathlib import Path

from metrics.quality import compute_quality_loss, select_primary_loss
from run_util.data_utils import load_requests, validate_requests_for_quality_mode


def test_code_task_uses_f1_as_primary_loss() -> None:
    assert select_primary_loss("code", strict=True) == "f1"
    loss, metrics = compute_quality_loss("code", "def f(): return 1", "def f(): return 1")
    assert loss == metrics["f1"] == 0.0


def test_baseline_quality_accepts_code_requests_with_reference(tmp_path: Path) -> None:
    request_path = tmp_path / "code_fixture.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "request_id": "code-001",
                "task": "code",
                "prompt": "Write a function",
                "reference": "def f():\n    return 1",
                "metadata": {
                    "source": "external_labeling",
                    "source_dataset": "longbench_lcc",
                    "split": "eval",
                    "risk_family": "low_risk",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    requests, _ = load_requests({"data": {"requests": str(request_path)}})

    validate_requests_for_quality_mode(requests, "baseline", {"full_gpu"})

    assert requests[0].task == "code"
