from __future__ import annotations

import sys
from pathlib import Path

import pytest

LABELING_SCRIPTS = Path("/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/scripts")
if str(LABELING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LABELING_SCRIPTS))


from prepare_sharegpt_session_candidates import build_session_candidates
from label_sharegpt_sessions import build_fixture


def _row(session_id: str, turn: int, prompt: str) -> dict:
    return {
        "request_id": f"{session_id}_t{turn:03d}",
        "session_id": session_id,
        "prompt": prompt,
        "reference": f"ref-{turn}",
        "source_turn_index": turn,
        "dataset_config": "sharegpt",
        "category": "test",
    }


def test_build_session_candidates_truncates_long_prompts() -> None:
    rows = [
        _row("s1", 0, "short"),
        _row("s1", 1, "x" * 6000),
        _row("s1", 2, "mid"),
        _row("s1", 3, "tail"),
    ]

    candidates = build_session_candidates(
        rows,
        max_sessions=1,
        min_turns=4,
        max_turns=4,
        max_prompt_chars=2048,
    )

    assert len(candidates) == 4
    assert max(len(row["prompt"]) for row in candidates) == 2048
    assert candidates[1]["metadata"]["length_bucket"] in {"long", "xl"}


def test_build_session_candidates_keeps_session_shape_after_truncation() -> None:
    rows = []
    for turn in range(4):
        rows.append(_row("s1", turn, "a" * (3000 if turn == 2 else 20)))
        rows.append(_row("s2", turn, "b" * 10))

    candidates = build_session_candidates(
        rows,
        max_sessions=2,
        min_turns=4,
        max_turns=4,
        max_prompt_chars=2048,
    )

    session_turns: dict[str, list[int]] = {}
    for row in candidates:
        session_turns.setdefault(str(row["session_id"]), []).append(int(row["turn_index"]))

    assert session_turns == {"s1": [0, 1, 2, 3], "s2": [0, 1, 2, 3]}


def test_build_session_candidates_filters_sessions_by_effective_prompt_budget() -> None:
    rows = [
        _row("safe", 0, "a" * 100),
        _row("safe", 1, "b" * 100),
        _row("safe", 2, "c" * 100),
        _row("safe", 3, "d" * 100),
        _row("unsafe", 0, "u" * 1200),
        _row("unsafe", 1, "v" * 1200),
        _row("unsafe", 2, "w" * 1200),
        _row("unsafe", 3, "x" * 1200),
    ]

    candidates = build_session_candidates(
        rows,
        max_sessions=1,
        min_turns=4,
        max_turns=4,
        max_prompt_chars=1024,
        max_effective_prompt_chars=2500,
    )

    assert {row["session_id"] for row in candidates} == {"safe"}


def test_build_fixture_rejects_legacy_sessions_without_hybrid_provenance() -> None:
    candidate_rows = []
    measurements = []
    for session_index in range(2):
        session_id = f"s{session_index}"
        for turn_index in range(4):
            candidate_rows.append(
                {
                    "request_id": f"{session_id}_t{turn_index}",
                    "session_id": session_id,
                    "prompt": f"prompt-{session_index}-{turn_index}",
                    "reference": f"ref-{session_index}-{turn_index}",
                    "turn_index": turn_index,
                    "metadata": {"source_dataset": "sharegpt_pilot", "split": "candidate", "risk_family": "unlabeled"},
                }
            )
            measurements.append(
                type(
                    "Row",
                    (),
                    {
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "quality_loss": 0.0,
                        "profile": "kivi_4bit_residual32",
                    },
                )()
            )

    with pytest.raises(ValueError, match="invalid hybrid provenance"):
        build_fixture(candidate_rows, measurements)


def test_build_fixture_rejects_legacy_four_turn_sessions() -> None:
    candidate_rows = []
    measurements = []
    for session_index in range(4):
        session_id = f"s{session_index}"
        for turn_index in range(4):
            candidate_rows.append(
                {
                    "request_id": f"{session_id}_t{turn_index}",
                    "session_id": session_id,
                    "prompt": f"prompt-{session_index}-{turn_index}",
                    "reference": f"ref-{session_index}-{turn_index}",
                    "turn_index": turn_index,
                    "metadata": {"source_dataset": "sharegpt_pilot", "split": "candidate", "risk_family": "unlabeled"},
                }
            )
            measurements.append(
                type(
                    "Row",
                    (),
                    {
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "quality_loss": 0.0,
                        "profile": "kivi_4bit_residual32",
                    },
                )()
            )

    with pytest.raises(ValueError, match="invalid hybrid provenance"):
        build_fixture(candidate_rows, measurements)
