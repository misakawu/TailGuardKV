#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_QUALITY_DEST = REPO_ROOT / "data" / "fixtures" / "baseline_quality_external.jsonl"
DEFAULT_BASELINE_SESSION_DEST = REPO_ROOT / "data" / "fixtures" / "baseline_session_external.jsonl"

QUALITY_REQUIRED_TOP_LEVEL = ("request_id", "task", "prompt", "reference", "metadata")
SESSION_REQUIRED_TOP_LEVEL = ("request_id", "task", "prompt", "reference", "session_id", "turn_index", "arrival_index", "metadata")
REQUIRED_METADATA = ("source", "source_dataset", "split", "risk_family")
ALLOWED_QUALITY_TASKS = {"qa", "summary", "code"}
REQUIRED_RISK_FAMILIES = {"low_risk"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import external baseline fixtures into the repo.")
    parser.add_argument("--kind", choices=("baseline_quality", "baseline_session"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.input)
    if args.kind == "baseline_quality":
        report = validate_baseline_quality_fixture(source_path)
        destination = Path(args.output) if args.output else DEFAULT_BASELINE_QUALITY_DEST
    else:
        report = validate_baseline_session_fixture(source_path)
        destination = Path(args.output) if args.output else DEFAULT_BASELINE_SESSION_DEST

    if args.validate_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    print(
        json.dumps(
            {
                "kind": args.kind,
                "input": str(source_path),
                "output": str(destination),
                "report": report,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_baseline_quality_fixture(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"baseline_quality 夹具为空: {path}")

    tasks: set[str] = set()
    splits: set[str] = set()
    risk_families: set[str] = set()
    for index, row in enumerate(rows):
        _require_fields(row, QUALITY_REQUIRED_TOP_LEVEL, path=path, row_index=index)
        metadata = _require_metadata(row, path=path, row_index=index)
        task = str(row["task"]).strip().lower()
        if task not in ALLOWED_QUALITY_TASKS:
            raise ValueError(f"baseline_quality task 不支持: row={index} task={task}")
        if not str(row["prompt"]).strip():
            raise ValueError(f"baseline_quality prompt 不能为空: row={index}")
        if not str(row["reference"]).strip():
            raise ValueError(f"baseline_quality reference 不能为空: row={index}")
        tasks.add(task)
        splits.add(str(metadata["split"]))
        risk_families.add(str(metadata["risk_family"]))

    if not {"calibration", "eval"} <= splits:
        raise ValueError(f"baseline_quality 必须同时包含 calibration 和 eval: got={sorted(splits)}")
    if not REQUIRED_RISK_FAMILIES <= risk_families:
        raise ValueError(f"baseline_quality 缺少风险组: need={sorted(REQUIRED_RISK_FAMILIES)} got={sorted(risk_families)}")
    if not tasks <= ALLOWED_QUALITY_TASKS:
        raise ValueError(f"baseline_quality task 非法: {sorted(tasks)}")

    return {
        "path": str(path),
        "row_count": len(rows),
        "tasks": sorted(tasks),
        "splits": sorted(splits),
        "risk_families": sorted(risk_families),
    }


def validate_baseline_session_fixture(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"baseline_session 夹具为空: {path}")

    sessions: dict[str, list[dict[str, Any]]] = {}
    splits: set[str] = set()
    risk_families: set[str] = set()
    arrivals: list[int] = []

    for index, row in enumerate(rows):
        _require_fields(row, SESSION_REQUIRED_TOP_LEVEL, path=path, row_index=index)
        metadata = _require_metadata(row, path=path, row_index=index)
        session_id = str(row["session_id"]).strip()
        if not session_id:
            raise ValueError(f"baseline_session session_id 不能为空: row={index}")
        turn_index = _parse_int(row["turn_index"], field_name="turn_index", row_index=index)
        arrival_index = _parse_int(row["arrival_index"], field_name="arrival_index", row_index=index)
        if arrival_index < 0:
            raise ValueError(f"baseline_session arrival_index 必须非负: row={index}")
        if not str(row["prompt"]).strip():
            raise ValueError(f"baseline_session prompt 不能为空: row={index}")
        sessions.setdefault(session_id, []).append({**row, "turn_index": turn_index, "arrival_index": arrival_index})
        splits.add(str(metadata["split"]))
        risk_families.add(str(metadata["risk_family"]))
        arrivals.append(arrival_index)

    if arrivals != sorted(arrivals):
        raise ValueError("baseline_session arrival_index 必须全局单调递增")
    if len(set(arrivals)) != len(arrivals):
        raise ValueError("baseline_session arrival_index 不能重复")

    multi_turn_session_count = 0
    for session_id, session_rows in sessions.items():
        ordered = sorted(session_rows, key=lambda row: (row["turn_index"], row["arrival_index"], str(row["request_id"])))
        expected_turn = 0
        previous_arrival = -1
        for row in ordered:
            if row["turn_index"] != expected_turn:
                raise ValueError(
                    "baseline_session turn_index 必须从 0 连续递增；"
                    f" session={session_id} expected={expected_turn} actual={row['turn_index']}"
                )
            if row["arrival_index"] < previous_arrival:
                raise ValueError(
                    "baseline_session arrival_index 必须随 turn_index 非递减；"
                    f" session={session_id} turn_index={row['turn_index']}"
                )
            previous_arrival = row["arrival_index"]
            expected_turn += 1
        if len(ordered) >= 2:
            multi_turn_session_count += 1

    interleaved_session_count = _count_interleaved_sessions(rows)

    if len(sessions) < 2:
        raise ValueError("baseline_session 至少需要两个 session")
    if multi_turn_session_count < 1:
        raise ValueError("baseline_session 至少需要一个多轮 session")
    if interleaved_session_count < 2:
        raise ValueError("baseline_session 至少需要两个可交错 session")
    if not {"calibration", "eval"} <= splits:
        raise ValueError(f"baseline_session 必须同时包含 calibration 和 eval: got={sorted(splits)}")
    if not REQUIRED_RISK_FAMILIES <= risk_families:
        raise ValueError(f"baseline_session 缺少风险组: need={sorted(REQUIRED_RISK_FAMILIES)} got={sorted(risk_families)}")

    return {
        "path": str(path),
        "row_count": len(rows),
        "session_count": len(sessions),
        "multi_turn_session_count": multi_turn_session_count,
        "interleaved_session_count": interleaved_session_count,
        "splits": sorted(splits),
        "risk_families": sorted(risk_families),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"夹具不存在: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        raise ValueError(f"仅支持数组 JSON: {path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _require_fields(row: dict[str, Any], fields: tuple[str, ...], *, path: Path, row_index: int) -> None:
    for field in fields:
        if field not in row:
            raise ValueError(f"缺少字段 {field}: path={path} row={row_index}")


def _require_metadata(row: dict[str, Any], *, path: Path, row_index: int) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata 必须是对象: path={path} row={row_index}")
    for key in REQUIRED_METADATA:
        value = metadata.get(key)
        if value in {None, ""}:
            raise ValueError(f"metadata 缺少 {key}: path={path} row={row_index}")
    return metadata


def _parse_int(value: Any, *, field_name: str, row_index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数: row={row_index}") from exc


def _count_interleaved_sessions(rows: list[dict[str, Any]]) -> int:
    ordered = sorted(rows, key=lambda row: int(row["arrival_index"]))
    interleaved: set[str] = set()
    for left, middle, right in zip(ordered, ordered[1:], ordered[2:]):
        left_session = str(left["session_id"])
        middle_session = str(middle["session_id"])
        right_session = str(right["session_id"])
        if left_session == right_session and left_session != middle_session:
            interleaved.add(left_session)
            interleaved.add(middle_session)
    return len(interleaved)


if __name__ == "__main__":
    raise SystemExit(main())
