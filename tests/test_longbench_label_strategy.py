from __future__ import annotations

import sys
from pathlib import Path


LABELING_SCRIPTS = Path("/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts")
if str(LABELING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LABELING_SCRIPTS))


from label_longbench_quality import compute_selection_targets, select_balanced_quality_rows


def _row(request_id: str, *, task: str, risk_family: str, prompt_len: int = 100) -> dict:
    return {
        "request_id": request_id,
        "task": task,
        "prompt": "x" * prompt_len,
        "reference": "ref",
        "metadata": {
            "source": "external_labeling_workspace",
            "source_dataset": f"longbench_{task}",
            "risk_family": risk_family,
        },
    }


def test_compute_selection_targets_uses_tie_pool_to_expand_sensitive_quota() -> None:
    targets = compute_selection_targets(
        strict_counts={
            "kivi_sensitive": 11,
            "h2o_sensitive": 11,
            "low_risk": 542,
        },
        tie_sensitive_count=36,
    )

    assert targets == {
        "kivi_sensitive": 29,
        "h2o_sensitive": 29,
        "low_risk": 60,
    }


def test_select_balanced_quality_rows_backfills_from_tie_pool_and_preserves_summary_floor() -> None:
    strict_rows = []
    for index in range(11):
        strict_rows.append(_row(f"kivi-{index}", task="qa", risk_family="kivi_sensitive"))
        strict_rows.append(_row(f"h2o-{index}", task="code", risk_family="h2o_sensitive"))

    tie_rows = []
    for index in range(18):
        tie_rows.append(_row(f"tie-qa-{index}", task="qa", risk_family="tie_sensitive"))
        tie_rows.append(_row(f"tie-code-{index}", task="code", risk_family="tie_sensitive"))

    low_risk_rows = []
    for index in range(20):
        low_risk_rows.append(_row(f"sum-{index}", task="summary", risk_family="low_risk"))
    for index in range(20):
        low_risk_rows.append(_row(f"qa-low-{index}", task="qa", risk_family="low_risk"))
    for index in range(20):
        low_risk_rows.append(_row(f"code-low-{index}", task="code", risk_family="low_risk"))

    selected = select_balanced_quality_rows(strict_rows + tie_rows + low_risk_rows)

    risk_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in selected:
        risk = str(row["metadata"]["risk_family"])
        task = str(row["task"])
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1

    assert risk_counts == {
        "kivi_sensitive": 29,
        "h2o_sensitive": 29,
        "low_risk": 60,
    }
    assert task_counts["summary"] >= 20
