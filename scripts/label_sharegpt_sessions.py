#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    BASELINE_SESSION_FIXTURE,
    BASELINE_SESSION_MANIFEST,
    LOW_RISK_THRESHOLD,
    MAINSTREAM_H2O_PROFILES,
    MAINSTREAM_KIVI_PROFILES,
    SENSITIVE_THRESHOLD,
    WORKSPACE_ROOT,
    ensure_artifact_dirs,
    ensure_repo_import_path,
    read_jsonl,
    write_json,
    write_jsonl,
)

ensure_repo_import_path()

from run_util.io_utils import read_measurements


HYBRID_CANDIDATES = WORKSPACE_ROOT / "artifacts/candidates/hybrid_session_candidates.jsonl"
HYBRID_MEASUREMENTS = WORKSPACE_ROOT / "artifacts/measurements/hybrid_session_candidates_profiles.csv"
RISK_FAMILIES = ("kivi_sensitive", "h2o_sensitive", "low_risk")
SESSION_TASKS = ("qa", "summary")
SESSIONS_PER_RISK_TASK = 8
CROSS_FAMILY_GAP = 0.02
HYBRID_BUILDER_SOURCE = "hybrid_session_builder"
HYBRID_SOURCE_DATASET = "sharegpt_longbench_hybrid_session"
FULL_GPU_PROFILE = "full_gpu"
EXPECTED_MAINSTREAM_PROFILE_ORDER = (
    FULL_GPU_PROFILE,
    *MAINSTREAM_KIVI_PROFILES,
    *MAINSTREAM_H2O_PROFILES,
)
EXPECTED_MAINSTREAM_PROFILES = frozenset(EXPECTED_MAINSTREAM_PROFILE_ORDER)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label hybrid session measurements and export baseline_session fixture.")
    parser.add_argument("--candidates", default=str(HYBRID_CANDIDATES))
    parser.add_argument("--measurements", default=str(HYBRID_MEASUREMENTS))
    parser.add_argument("--output", default=str(BASELINE_SESSION_FIXTURE))
    parser.add_argument("--manifest", default=str(BASELINE_SESSION_MANIFEST))
    args = parser.parse_args()

    ensure_artifact_dirs()
    candidate_rows = read_jsonl(Path(args.candidates))
    measurements = read_measurements(Path(args.measurements))
    fixture, manifest = build_fixture(candidate_rows, measurements)
    write_jsonl(Path(args.output), fixture)
    write_json(Path(args.manifest), manifest)
    print(json.dumps({"output": args.output, "rows": len(fixture)}, ensure_ascii=False))
    return 0


def build_fixture(candidate_rows: list[dict[str, Any]], measurements: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reserves = select_hybrid_candidate_reserves(candidate_rows, measurements, required_per_cell=12)
    selected_sessions = select_balanced_sessions(reserves)
    exported = interleave_sessions(selected_sessions)
    manifest = {
        "schema_version": 1,
        "fixture_hash": _fixture_hash(exported),
        "rows": len(exported),
        "sessions": len({row["session_id"] for row in exported}),
        "risk_distribution": _count_by(exported, "risk_family"),
        "splits": _count_split(exported),
        "session_tasks": _count_session_tasks(exported),
        "risk_task_split_sessions": _count_risk_task_split_sessions(exported),
        "candidate_reserve": _reserve_provenance(reserves),
        "session_provenance": _selected_session_provenance(selected_sessions),
        "profile_coverage": {
            "profiles": list(EXPECTED_MAINSTREAM_PROFILE_ORDER),
            "content_turn_measurements_per_profile": len(selected_sessions) * 3,
            "complete_sessions": len(selected_sessions),
        },
    }
    return exported, manifest


def _fixture_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_hybrid_candidate_reserves(
    candidate_rows: list[dict[str, Any]],
    measurements: list[Any],
    *,
    required_per_cell: int = 12,
    source_exhausted: bool = False,
) -> list[tuple[str, str, str, list[dict[str, Any]], float]]:
    """Keep only final-form, fully measured candidates for later fixture selection."""
    if required_per_cell <= 0:
        raise ValueError("required_per_cell must be positive")
    grouped: dict[tuple[str, str], list[tuple[str, str, str, list[dict[str, Any]], float]]] = defaultdict(list)
    for item in _labeled_hybrid_sessions(candidate_rows, measurements):
        grouped[(item[0], item[1])].append(item)

    selected: list[tuple[str, str, str, list[dict[str, Any]], float]] = []
    for risk_family in RISK_FAMILIES:
        for task in SESSION_TASKS:
            pool = grouped[(risk_family, task)]
            if len(pool) < required_per_cell:
                raise RuntimeError(
                    "hybrid candidate cell insufficient: "
                    f"{risk_family}/{task} available={len(pool)} required={required_per_cell} "
                    f"source_exhausted={source_exhausted}"
                )
            selected.extend(sorted(pool, key=lambda item: (-item[4], item[2]))[:required_per_cell])
    return selected


def strict_hybrid_cell_counts(
    candidate_rows: list[dict[str, Any]], measurements: list[Any]
) -> dict[tuple[str, str], int]:
    counts = {(risk_family, task): 0 for risk_family in RISK_FAMILIES for task in SESSION_TASKS}
    for risk_family, task, _, _, _ in _labeled_hybrid_sessions(candidate_rows, measurements):
        counts[(risk_family, task)] += 1
    return counts


def _labeled_hybrid_sessions(
    candidate_rows: list[dict[str, Any]], measurements: list[Any]
) -> list[tuple[str, str, str, list[dict[str, Any]], float]]:
    candidate_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in candidate_rows:
        _validate_hybrid_candidate_row(row)
        key = _candidate_key(row)
        if key in candidate_by_key:
            raise ValueError(f"duplicate hybrid candidate key: {key}")
        candidate_by_key[key] = row
        candidate_by_session[str(row["session_id"])].append(row)
    _validate_unique_session_sources(candidate_by_session)

    measurement_by_key: dict[tuple[str, str, int], dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in measurements:
        key = (
            str(getattr(row, "request_id", "")),
            str(getattr(row, "session_id", "") or ""),
            int(getattr(row, "turn_index", -1)),
        )
        if key not in candidate_by_key:
            raise ValueError(
                "measurement does not match a hybrid candidate row: "
                f"request_id={key[0]} session_id={key[1]} turn_index={key[2]}"
            )
        profile = str(getattr(row, "profile", ""))
        if profile not in EXPECTED_MAINSTREAM_PROFILES:
            continue
        measurement_by_key[key][profile].append(row)

    labeled_sessions: list[tuple[str, str, str, list[dict[str, Any]], float]] = []
    for session_id, session_rows in candidate_by_session.items():
        ordered = _complete_session_rows(session_rows)
        if ordered is None:
            raise ValueError(f"hybrid candidate session is not a complete five-turn session: {session_id}")
        session_task = str(ordered[2].get("task", "")).lower()
        if session_task not in SESSION_TASKS:
            continue
        profile_losses = _validated_session_profile_losses(ordered, measurement_by_key)
        risk_family = classify_profile_losses(profile_losses)
        if risk_family is None:
            continue
        labeled_sessions.append(
            (risk_family, session_task, session_id, ordered, _risk_rank(risk_family, profile_losses))
        )

    return labeled_sessions


def classify_profile_losses(profile_losses: dict[str, float]) -> str | None:
    if not EXPECTED_MAINSTREAM_PROFILES.issubset(profile_losses):
        return None
    kivi_max = max(profile_losses[profile] for profile in MAINSTREAM_KIVI_PROFILES)
    h2o_max = max(profile_losses[profile] for profile in MAINSTREAM_H2O_PROFILES)
    overall_max = max(kivi_max, h2o_max)
    if overall_max <= LOW_RISK_THRESHOLD:
        return "low_risk"
    if kivi_max >= SENSITIVE_THRESHOLD and kivi_max - h2o_max >= CROSS_FAMILY_GAP:
        return "kivi_sensitive"
    if h2o_max >= SENSITIVE_THRESHOLD and h2o_max - kivi_max >= CROSS_FAMILY_GAP:
        return "h2o_sensitive"
    return None


def select_balanced_sessions(
    labeled_sessions: list[tuple[str, str, str, list[dict[str, Any]], float]],
) -> list[tuple[str, str, str, str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[tuple[str, str, str, list[dict[str, Any]], float]]] = defaultdict(list)
    for item in labeled_sessions:
        grouped[(item[0], item[1])].append(item)

    selected: list[tuple[str, str, str, str, list[dict[str, Any]]]] = []
    for risk_family in RISK_FAMILIES:
        for task in SESSION_TASKS:
            pool = grouped.get((risk_family, task), [])
            if len(pool) < SESSIONS_PER_RISK_TASK:
                raise RuntimeError(
                    f"{risk_family} {task} hybrid session pool insufficient: "
                    f"{len(pool)} < {SESSIONS_PER_RISK_TASK}"
                )
            ranked = sorted(
                pool,
                key=lambda item: (-item[4], _content_source_request_id(item[3]), item[2]),
            )[:SESSIONS_PER_RISK_TASK]
            for index, item in enumerate(ranked):
                split = "calibration" if index % 2 == 0 else "eval"
                selected.append((risk_family, task, split, item[2], item[3]))
    return selected


def interleave_sessions(
    sessions: list[tuple[str, str, str, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    session_payloads = sorted(sessions, key=lambda item: (item[2], item[0], item[1], item[3]))
    exported: list[dict[str, Any]] = []
    for turn_offset in range(5):
        for risk_family, _, split, _, rows in session_payloads:
            row = dict(rows[turn_offset])
            row["arrival_index"] = len(exported)
            row["metadata"] = {
                **row.get("metadata", {}),
                "risk_family": risk_family,
                "split": split,
            }
            exported.append(row)
    return exported


def _complete_session_rows(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    ordered = sorted(session_rows, key=lambda row: int(row["turn_index"]))
    if len(ordered) != 5 or [int(row["turn_index"]) for row in ordered] != list(range(5)):
        return None
    return ordered


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["request_id"]), str(row["session_id"]), int(row["turn_index"]))


def _validate_hybrid_candidate_row(row: dict[str, Any]) -> None:
    metadata = row.get("metadata") or {}
    required_metadata = (
        "content_source_dataset",
        "content_source_request_id",
        "content_source_index",
        "content_source_record_id",
        "content_payload_hash",
        "injection_template",
        "original_session_id",
    )
    provenance_valid = (
        metadata.get("source") == HYBRID_BUILDER_SOURCE
        and metadata.get("source_dataset") == HYBRID_SOURCE_DATASET
        and all(metadata.get(field) not in {None, ""} for field in required_metadata)
    )
    if int(row.get("turn_index", -1)) >= 2:
        provenance_valid = provenance_valid and metadata.get("content_source_dataset") in {"longbench", "raghot_qa"}
    if not provenance_valid:
        raise ValueError(
            "invalid hybrid provenance: "
            f"request_id={row.get('request_id', '')} session_id={row.get('session_id', '')}"
        )


def _validated_session_profile_losses(
    ordered_rows: list[dict[str, Any]],
    measurement_by_key: dict[tuple[str, str, int], dict[str, list[Any]]],
) -> dict[str, float]:
    profile_losses: dict[str, float] = {}
    for candidate_row in ordered_rows[2:5]:
        key = _candidate_key(candidate_row)
        profiles = measurement_by_key.get(key, {})
        missing_profiles = sorted(EXPECTED_MAINSTREAM_PROFILES - profiles.keys())
        if missing_profiles:
            raise RuntimeError(
                "hybrid measurement coverage incomplete: "
                f"session={key[1]} request_id={key[0]} turn={key[2]} "
                f"missing_profiles={missing_profiles}"
            )
        failed_profiles = sorted(
            profile
            for profile in EXPECTED_MAINSTREAM_PROFILES
            if any(
                not bool(getattr(row, "ok", False))
                or not bool(getattr(row, "measured", False))
                or getattr(row, "quality_loss", None) is None
                for row in profiles[profile]
            )
        )
        if failed_profiles:
            raise RuntimeError(
                "hybrid measurement coverage failed: "
                f"session={key[1]} request_id={key[0]} turn={key[2]} "
                f"failed_profiles={failed_profiles}"
            )
        for profile in EXPECTED_MAINSTREAM_PROFILES:
            turn_loss = max(float(row.quality_loss) for row in profiles[profile])
            profile_losses[profile] = max(turn_loss, profile_losses.get(profile, turn_loss))
    return profile_losses


def _validate_unique_session_sources(candidate_by_session: dict[str, list[dict[str, Any]]]) -> None:
    seen_skeletons: set[str] = set()
    seen_content: set[tuple[str, str]] = set()
    seen_payloads: set[str] = set()
    for session_id, rows in candidate_by_session.items():
        ordered = _complete_session_rows(rows)
        if ordered is None:
            continue
        _validate_complete_hybrid_session(session_id, ordered)
        metadata_rows = [row.get("metadata") or {} for row in ordered]
        fields = (
            "source",
            "source_dataset",
            "injection_template",
            "original_session_id",
            "content_source_dataset",
            "content_source_request_id",
            "content_source_record_id",
            "content_payload_hash",
        )
        for field in fields:
            if len({str(metadata.get(field, "")) for metadata in metadata_rows}) != 1:
                raise ValueError(f"inconsistent {field} within session: {session_id}")
        metadata = metadata_rows[0]
        original_session_id = str(metadata.get("original_session_id", ""))
        content_key = (
            str(metadata.get("content_source_dataset", "")),
            str(metadata.get("content_source_record_id", "")),
        )
        payload_hash = str(metadata.get("content_payload_hash", ""))
        if not original_session_id or original_session_id in seen_skeletons:
            raise ValueError(f"duplicate original_session_id: {original_session_id}")
        if not all(content_key) or content_key in seen_content:
            raise ValueError(f"duplicate content source record: {content_key}")
        if not payload_hash or payload_hash in seen_payloads:
            raise ValueError(f"duplicate injected payload: {payload_hash}")
        seen_skeletons.add(original_session_id)
        seen_content.add(content_key)
        seen_payloads.add(payload_hash)


def _validate_complete_hybrid_session(session_id: str, ordered: list[dict[str, Any]]) -> None:
    metadata_rows = [row.get("metadata") or {} for row in ordered]
    roles = [str(metadata.get("hybrid_turn_role", "")) for metadata in metadata_rows]
    expected_roles = [
        "sharegpt_opening",
        "sharegpt_opening",
        "content_query",
        "reference_recall",
        "reference_rewrite",
    ]
    if roles != expected_roles:
        raise ValueError(f"invalid hybrid session roles: {session_id}")
    if [bool(metadata.get("last_turn", False)) for metadata in metadata_rows] != [False, False, False, False, True]:
        raise ValueError(f"invalid hybrid session last_turn provenance: {session_id}")
    content_rows = ordered[2:]
    content_tasks = {str(row.get("task", "")).lower() for row in content_rows}
    if len(content_tasks) != 1 or not content_tasks <= set(SESSION_TASKS):
        raise ValueError(f"inconsistent hybrid content task: {session_id}")
    content_references = {str(row.get("reference", "")) for row in content_rows}
    if len(content_references) != 1:
        raise ValueError(f"inconsistent hybrid content reference: {session_id}")
    content_dataset = str(metadata_rows[0].get("content_source_dataset", ""))
    if content_dataset not in {"longbench", "raghot_qa"}:
        raise ValueError(f"invalid hybrid content source dataset: {session_id}")
    if content_dataset == "raghot_qa":
        if content_tasks != {"qa"}:
            raise ValueError(f"RAGhot Summary content is not permitted: {session_id}")
        for field in ("context_pack_hash", "supporting_fact_ids", "packing_policy_version"):
            values = [metadata.get(field) for metadata in metadata_rows]
            if not all(values) or len({json.dumps(value, sort_keys=True) for value in values}) != 1:
                raise ValueError(f"inconsistent RAGhot {field} within session: {session_id}")


def _risk_rank(risk_family: str, profile_losses: dict[str, float]) -> float:
    kivi_max = max(profile_losses[profile] for profile in MAINSTREAM_KIVI_PROFILES)
    h2o_max = max(profile_losses[profile] for profile in MAINSTREAM_H2O_PROFILES)
    if risk_family == "kivi_sensitive":
        return kivi_max - h2o_max
    if risk_family == "h2o_sensitive":
        return h2o_max - kivi_max
    return -max(kivi_max, h2o_max)


def _content_source_request_id(rows: list[dict[str, Any]]) -> str:
    return str((rows[0].get("metadata") or {}).get("content_source_request_id", ""))


def _session_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = rows[0].get("metadata") or {}
    return {
        "source": str(metadata.get("content_source_dataset", "")),
        "original_session_id": str(metadata.get("original_session_id", "")),
        "content_id": {
            "request_id": str(metadata.get("content_source_request_id", "")),
            "record_id": str(metadata.get("content_source_record_id", "")),
        },
        "payload_hash": str(metadata.get("content_payload_hash", "")),
    }


def _reserve_provenance(
    reserves: list[tuple[str, str, str, list[dict[str, Any]], float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for risk_family, task, session_id, rows, risk_margin in reserves:
        records.append(
            {
                "risk_family": risk_family,
                "task": task,
                "session_id": session_id,
                "risk_margin": round(risk_margin, 6),
                **_session_provenance(rows),
            }
        )
    return sorted(records, key=lambda row: (row["risk_family"], row["task"], row["session_id"]))


def _selected_session_provenance(
    sessions: list[tuple[str, str, str, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for risk_family, task, split, session_id, rows in sessions:
        records.append(
            {
                "risk_family": risk_family,
                "task": task,
                "split": split,
                "session_id": session_id,
                **_session_provenance(rows),
            }
        )
    return sorted(records, key=lambda row: (row["risk_family"], row["task"], row["split"], row["session_id"]))


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get("split", ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_session_tasks(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if int(row["turn_index"]) != 2:
            continue
        task = str(row["task"])
        counts[task] = counts.get(task, 0) + 1
    return counts


def _count_risk_task_split_sessions(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if int(row["turn_index"]) != 2:
            continue
        metadata = row.get("metadata") or {}
        key = f"{metadata.get('risk_family', '')}/{row['task']}/{metadata.get('split', '')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
