from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_canonical_session_supervisor import (
    CanonicalSupervisorError,
    ensure_complete_replay,
    load_input_rows,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"request_id": f"s{session}_t{turn}", "session_id": f"s{session}", "turn_index": turn,
         "arrival_index": session * 5 + turn, "task": "chat", "prompt": "short", "reference": "reference", "metadata": {}}
        for session in range(6) for turn in range(5)
    ]


def test_supervisor_rejects_truncated_probe_input(tmp_path: Path) -> None:
    input_path = tmp_path / "probe.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in _rows()), encoding="utf-8")

    with pytest.raises(CanonicalSupervisorError, match="never truncates"):
        load_input_rows(input_path, expected_requests=30, max_requests=29)


def test_supervisor_refuses_merge_when_profile_is_missing() -> None:
    rows = [{"request_id": "s0_t0", "profile": "full_gpu", "ok": True, "measured": True}]

    with pytest.raises(CanonicalSupervisorError, match="missing profiles"):
        ensure_complete_replay(rows, request_ids={"s0_t0"}, profiles=("full_gpu", "kivi_4bit_residual32"))
