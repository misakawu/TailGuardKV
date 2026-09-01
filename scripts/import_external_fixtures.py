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
HYBRID_SOURCE = "hybrid_session_builder"
HYBRID_SOURCE_DATASET = "sharegpt_longbench_hybrid_session"
HYBRID_REQUIRED_METADATA = (
    "content_source_dataset",
    "content_source_request_id",
    "content_source_index",
    "injection_template",
    "original_session_id",
    "hybrid_turn_role",
)
SESSION_RISK_FAMILIES = ("kivi_sensitive", "h2o_sensitive", "low_risk")
SESSION_TASKS = ("qa", "summary")
SESSION_SPLITS = ("calibration", "eval")
SESSION_COUNT = 48
TURNS_PER_SESSION = 5
SESSIONS_PER_RISK_TASK_SPLIT = 4
TURN_ROLES = ("sharegpt_opening", "sharegpt_opening", "longbench_content", "reference_recall", "reference_rewrite")


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
    request_ids: set[str] = set()

    for index, row in enumerate(rows):
        _require_fields(row, SESSION_REQUIRED_TOP_LEVEL, path=path, row_index=index)
        metadata = _require_metadata(row, path=path, row_index=index)
        request_id = str(row["request_id"]).strip()
        if not request_id:
            raise ValueError(f"baseline_session request_id 不能为空: row={index}")
        if request_id in request_ids:
            raise ValueError(f"baseline_session request_id 不能重复: row={index} request_id={request_id}")
        request_ids.add(request_id)
        session_id = str(row["session_id"]).strip()
        if not session_id:
            raise ValueError(f"baseline_session session_id 不能为空: row={index}")
        turn_index = _parse_int(row["turn_index"], field_name="turn_index", row_index=index)
        arrival_index = _parse_int(row["arrival_index"], field_name="arrival_index", row_index=index)
        _validate_hybrid_session_row(row, metadata, turn_index=turn_index, row_index=index)
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

    if arrivals != list(range(len(rows))):
        raise ValueError("baseline_session arrival_index 必须从 0 全局连续递增")
    if len(rows) != SESSION_COUNT * TURNS_PER_SESSION:
        raise ValueError(
            f"baseline_session 必须包含 {SESSION_COUNT * TURNS_PER_SESSION} 行: got={len(rows)}"
        )
    if len(sessions) != SESSION_COUNT:
        raise ValueError(f"baseline_session 必须包含 {SESSION_COUNT} 个 session: got={len(sessions)}")

    risk_task_split_session_counts: dict[str, int] = {}
    for session_id, session_rows in sessions.items():
        ordered = sorted(session_rows, key=lambda row: (row["turn_index"], row["arrival_index"], str(row["request_id"])))
        actual_turns = [row["turn_index"] for row in ordered]
        if actual_turns != list(range(TURNS_PER_SESSION)):
            raise ValueError(
                "baseline_session 每个 session 必须包含从 0 到 4 的连续 turn_index；"
                f" session={session_id} actual={actual_turns}"
            )

        metadata_rows = [row["metadata"] for row in ordered]
        session_splits = {str(metadata["split"]) for metadata in metadata_rows}
        session_risks = {str(metadata["risk_family"]) for metadata in metadata_rows}
        original_session_ids = {str(metadata["original_session_id"]) for metadata in metadata_rows}
        if len(session_splits) != 1 or len(session_risks) != 1 or len(original_session_ids) != 1:
            raise ValueError(
                "baseline_session split、risk_family 和 original_session_id 必须在 session 内保持一致；"
                f" session={session_id}"
            )

        opening_tasks = {str(row["task"]).strip().lower() for row in ordered[:2]}
        injected_tasks = {str(row["task"]).strip().lower() for row in ordered[2:]}
        if opening_tasks != {"chat"}:
            raise ValueError(f"baseline_session turn0/turn1 必须是 chat 开场: session={session_id}")
        if len(injected_tasks) != 1 or not injected_tasks <= set(SESSION_TASKS):
            raise ValueError(f"baseline_session 风险记录只能是 QA/Summary: session={session_id}")

        injected_metadata = metadata_rows[2:]
        if any(str(metadata["content_source_dataset"]).strip().lower() != "longbench" for metadata in injected_metadata):
            raise ValueError(f"baseline_session 注入 turn 缺少 LongBench provenance: session={session_id}")
        provenance_keys = ("content_source_request_id", "content_source_index", "injection_template")
        for key in provenance_keys:
            if len({str(metadata[key]) for metadata in injected_metadata}) != 1:
                raise ValueError(f"baseline_session 注入 provenance 在 session 内不一致: session={session_id} field={key}")
        if len({str(row["reference"]) for row in ordered[2:]}) != 1:
            raise ValueError(f"baseline_session turn2/turn3/turn4 必须复用 LongBench reference: session={session_id}")

        risk_family = next(iter(session_risks))
        split = next(iter(session_splits))
        task = next(iter(injected_tasks))
        key = f"{risk_family}/{task}/{split}"
        risk_task_split_session_counts[key] = risk_task_split_session_counts.get(key, 0) + 1

    expected_counts = {
        f"{risk_family}/{task}/{split}": SESSIONS_PER_RISK_TASK_SPLIT
        for risk_family in SESSION_RISK_FAMILIES
        for task in SESSION_TASKS
        for split in SESSION_SPLITS
    }
    if risk_task_split_session_counts != expected_counts:
        raise ValueError(
            "baseline_session 每个 risk/task/split 必须包含 4 个 session: "
            f"got={dict(sorted(risk_task_split_session_counts.items()))}"
        )

    expected_turn_order = [turn_index for turn_index in range(TURNS_PER_SESSION) for _ in range(SESSION_COUNT)]
    if [_parse_int(row["turn_index"], field_name="turn_index", row_index=index) for index, row in enumerate(rows)] != expected_turn_order:
        raise ValueError("baseline_session 必须按 turn 全局交错")

    return {
        "path": str(path),
        "row_count": len(rows),
        "session_count": len(sessions),
        "turns_per_session": TURNS_PER_SESSION,
        "multi_turn_session_count": len(sessions),
        "interleaved_session_count": len(sessions),
        "splits": sorted(splits),
        "risk_families": sorted(risk_families),
        "risk_task_split_session_counts": dict(sorted(risk_task_split_session_counts.items())),
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


def _validate_hybrid_session_row(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    turn_index: int,
    row_index: int,
) -> None:
    if metadata.get("source") != HYBRID_SOURCE:
        raise ValueError(f"baseline_session source 必须是 {HYBRID_SOURCE}: row={row_index}")
    if metadata.get("source_dataset") != HYBRID_SOURCE_DATASET:
        raise ValueError(f"baseline_session source_dataset 必须是 {HYBRID_SOURCE_DATASET}: row={row_index}")
    for key in HYBRID_REQUIRED_METADATA:
        value = metadata.get(key)
        if value is None or value == "":
            raise ValueError(f"baseline_session metadata 缺少 {key}: row={row_index}")
    if str(metadata["split"]) not in SESSION_SPLITS:
        raise ValueError(f"baseline_session split 非法: row={row_index} split={metadata['split']}")
    if str(metadata["risk_family"]) not in SESSION_RISK_FAMILIES:
        raise ValueError(f"baseline_session risk_family 非法: row={row_index} risk_family={metadata['risk_family']}")
    if turn_index < 0 or turn_index >= TURNS_PER_SESSION:
        raise ValueError(f"baseline_session turn_index 必须在 0..4: row={row_index} turn_index={turn_index}")
    if str(metadata["hybrid_turn_role"]) != TURN_ROLES[turn_index]:
        raise ValueError(
            "baseline_session hybrid_turn_role 与 turn_index 不匹配: "
            f"row={row_index} role={metadata['hybrid_turn_role']} turn_index={turn_index}"
        )
    task = str(row["task"]).strip().lower()
    if turn_index < 2 and task != "chat":
        raise ValueError(f"baseline_session turn0/turn1 必须是 chat 开场: row={row_index}")
    if turn_index >= 2:
        if task not in SESSION_TASKS:
            raise ValueError(f"baseline_session 风险记录只能是 QA/Summary: row={row_index}")
        if str(metadata["content_source_dataset"]).strip().lower() != "longbench":
            raise ValueError(f"baseline_session 注入 turn 缺少 LongBench provenance: row={row_index}")
        if not str(row["reference"]).strip():
            raise ValueError(f"baseline_session 注入 turn reference 不能为空: row={row_index}")


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
