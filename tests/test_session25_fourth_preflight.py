from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.preflight_session25_fourth import audit_fixture, audit_tight_budget_rows


def _fixture(path: Path, *, session_count: int = 25, duplicate: bool = False) -> None:
    rows = []
    for arrival_index in range(session_count * 5):
        session_index, turn_index = divmod(arrival_index, 5)
        rows.append({
            "request_id": f"r{arrival_index if not duplicate else 0}",
            "session_id": f"s{session_index}",
            "turn_index": turn_index,
            "arrival_index": arrival_index,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_accepts_exactly_twenty_five_complete_sessions(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    _fixture(path)
    assert audit_fixture(path)["request_count"] == 125


def test_accepts_excluded_batch_arrival_gaps_without_reordering(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    _fixture(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[60:]:
        row["arrival_index"] += 10
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert audit_fixture(path)["arrival_gap_count"] == 10


def test_rejects_reordered_arrivals(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    _fixture(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[60]["arrival_index"] = rows[59]["arrival_index"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="arrival_index"):
        audit_fixture(path)


@pytest.mark.parametrize("session_count,duplicate", [(27, False), (25, True)])
def test_rejects_wrong_coverage_or_duplicate_requests(tmp_path: Path, session_count: int, duplicate: bool) -> None:
    path = tmp_path / "fixture.jsonl"
    _fixture(path, session_count=session_count, duplicate=duplicate)
    with pytest.raises(ValueError):
        audit_fixture(path)


def test_tight_budget_audit_requires_each_lossy_policy_and_budget_to_show_pressure() -> None:
    rows = []
    for policy in ("utility_dynamic", "uncalibrated_dynamic"):
        for budget in (10.0, 20.0):
            rows.append({"policy": policy, "memory_budget_mib": str(budget), "policy_budget_filtered": "true",
                         "backend_budget_hit": "false", "global_resident_kv_mib": str(budget)})
    report = audit_tight_budget_rows(rows, budgets=[10.0, 20.0], policies=["full_lru", "static_safe", "utility_dynamic", "uncalibrated_dynamic"])
    assert report["passed"] is True
    rows[-1]["global_resident_kv_mib"] = "21"
    assert audit_tight_budget_rows(rows, budgets=[10.0, 20.0], policies=["full_lru", "static_safe", "utility_dynamic", "uncalibrated_dynamic"])["passed"] is False
