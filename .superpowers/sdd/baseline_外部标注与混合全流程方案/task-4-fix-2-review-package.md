#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    LONG_BENCH_CANDIDATES,
    SHAREGPT_CANDIDATES,
    ensure_artifact_dirs,
    normalize_text,
    read_jsonl,
    write_json,
    write_jsonl,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
HYBRID_CANDIDATES = WORKSPACE_ROOT / "artifacts/candidates/hybrid_session_candidates.jsonl"
HYBRID_MANIFEST = WORKSPACE_ROOT / "artifacts/manifests/hybrid_session_candidates_manifest.json"
HYBRID_SKELETONS = WORKSPACE_ROOT / "artifacts/candidates/sharegpt_hybrid_skeletons.jsonl"
HYBRID_SOURCE = "sharegpt_longbench_hybrid_session"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic five-turn ShareGPT/LongBench hybrid sessions.")
    parser.add_argument("--sharegpt", default=str(HYBRID_SKELETONS))
    parser.add_argument("--longbench", default=str(LONG_BENCH_CANDIDATES))
    parser.add_argument("--output", default=str(HYBRID_CANDIDATES))
    parser.add_argument("--manifest", default=str(HYBRID_MANIFEST))
    parser.add_argument("--session-count", type=int, default=48)
    parser.add_argument("--max-content-prompt-chars", type=int, default=512)
    parser.add_argument("--task-counts", default="")
    parser.add_argument("--require-unique-sources", action="store_true")
    args = parser.parse_args()

    rows = build_hybrid_sessions(
        read_jsonl(Path(args.sharegpt)),
        read_jsonl(Path(args.longbench)),
        session_count=args.session_count,
        max_content_prompt_chars=args.max_content_prompt_chars,
        task_counts=json.loads(args.task_counts) if args.task_counts else None,
        require_unique_sources=args.require_unique_sources,
    )
    ensure_artifact_dirs()
    write_jsonl(Path(args.output), rows)
    write_json(
        Path(args.manifest),
        {
            "output": args.output,
            "rows": len(rows),
            "sessions": len({row["session_id"] for row in rows}),
            "tasks": _session_task_counts(rows),
            "turns_per_session": 5,
            "max_content_prompt_chars": args.max_content_prompt_chars,
        },
    )
    print(json.dumps({"output": args.output, "rows": len(rows)}, ensure_ascii=False))
    return 0


def build_hybrid_sessions(
    sharegpt_rows: list[dict[str, Any]],
    longbench_rows: list[dict[str, Any]],
    *,
    raghot_rows: list[dict[str, Any]] | None = None,
    use_raghot_qa: bool = False,
    session_count: int = 48,
    max_content_prompt_chars: int = 512,
    task_counts: dict[str, int] | None = None,
    require_unique_sources: bool = False,
) -> list[dict[str, Any]]:
    if session_count <= 0:
        raise ValueError("session_count must be positive")
    counts = _task_counts(session_count, task_counts)

    skeletons = _sharegpt_skeletons(sharegpt_rows, session_count=session_count)
    content_rows = _content_rows(
        longbench_rows,
        raghot_rows=raghot_rows,
        use_raghot_qa=use_raghot_qa,
        task_counts=counts,
    )
    sessions = [
        _build_session(index, skeleton, content, max_content_prompt_chars=max_content_prompt_chars)
        for index, (skeleton, content) in enumerate(zip(skeletons, content_rows, strict=True))
    ]
    if require_unique_sources:
        _validate_unique_final_sources(sessions)
    return _interleave_sessions(sessions)


def _sharegpt_skeletons(
    rows: list[dict[str, Any]],
    *,
    session_count: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, row in enumerate(rows):
        session_id = normalize_text(row.get("session_id"))
        if session_id:
            grouped[session_id].append((source_index, row))

    skeletons: list[list[tuple[int, dict[str, Any]]]] = []
    for session_id in sorted(grouped):
        ordered = sorted(grouped[session_id], key=lambda item: int(item[1].get("turn_index", 0)))
        if len(ordered) < 2:
            continue
        first_two = ordered[:2]
        if [int(item[1].get("turn_index", -1)) for item in first_two] != [0, 1]:
            continue
        if not all(normalize_text(item[1].get("prompt")) and normalize_text(item[1].get("reference")) for item in first_two):
            continue
        skeletons.append(first_two)
        if len(skeletons) == session_count:
            break
    if len(skeletons) < session_count:
        raise RuntimeError(f"ShareGPT hybrid skeleton pool insufficient: {len(skeletons)} < {session_count}")
    return skeletons


def _task_counts(session_count: int, task_counts: dict[str, int] | None) -> dict[str, int]:
    if task_counts is None:
        if session_count % 2:
            raise ValueError("session_count must be even for balanced task selection")
        return {"qa": session_count // 2, "summary": session_count // 2}
    normalized = {task: int(task_counts.get(task, 0)) for task in ("qa", "summary")}
    if any(count < 0 for count in normalized.values()) or sum(normalized.values()) != session_count:
        raise ValueError("task_counts must contain non-negative qa/summary counts totaling session_count")
    return normalized


def _content_rows(
    rows: list[dict[str, Any]],
    *,
    raghot_rows: list[dict[str, Any]] | None,
    use_raghot_qa: bool,
    task_counts: dict[str, int],
) -> list[tuple[int, dict[str, Any], str]]:
    by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {"qa": [], "summary": []}
    for source_index, row in enumerate(rows):
        task = normalize_text(row.get("task")).lower()
        if task not in by_task:
            continue
        if not normalize_text(row.get("prompt")) or not normalize_text(row.get("reference")):
            continue
        by_task[task].append((source_index, row))

    if use_raghot_qa:
        if raghot_rows is None:
            raise ValueError("RAGhot QA content was requested without RAGhot candidates")
        if any(normalize_text(row.get("task")).lower() != "qa" for row in raghot_rows):
            raise ValueError("RAGhot content supports QA only; Summary must use LongBench")
        raghot_qa = [
            (source_index, row)
            for source_index, row in enumerate(raghot_rows)
            if normalize_text(row.get("prompt")) and normalize_text(row.get("reference"))
        ]
        by_task["qa"] = raghot_qa

    for task in ("qa", "summary"):
        if len(by_task[task]) < task_counts[task]:
            dataset = "RAGhot QA" if task == "qa" and use_raghot_qa else "LongBench"
            raise RuntimeError(
                f"{dataset} {task} pool insufficient: {len(by_task[task])} < {task_counts[task]}"
            )

    selected: list[tuple[int, dict[str, Any], str]] = []
    qa_dataset = "raghot_qa" if use_raghot_qa else "longbench"
    for index in range(max(task_counts.values())):
        if index < task_counts["qa"]:
            source_index, row = by_task["qa"][index]
            selected.append((source_index, row, qa_dataset))
        if index < task_counts["summary"]:
            source_index, row = by_task["summary"][index]
            selected.append((source_index, row, "longbench"))
    return selected


def _build_session(
    session_index: int,
    skeleton: list[tuple[int, dict[str, Any]]],
    content: tuple[int, dict[str, Any], str],
    *,
    max_content_prompt_chars: int,
) -> list[dict[str, Any]]:
    session_id = f"hybrid-session-{session_index:03d}"
    original_session_id = normalize_text(skeleton[0][1].get("session_id"))
    content_index, content_row, content_dataset = content
    task = normalize_text(content_row.get("task")).lower()
    reference = normalize_text(content_row.get("reference"))
    content_prompt = _truncate_prompt(normalize_text(content_row.get("prompt")), max_content_prompt_chars)
    content_metadata = content_row.get("metadata") or {}
    content_provenance = {
        "content_source_dataset": content_dataset,
        "content_source_request_id": normalize_text(content_row.get("request_id")),
        "content_source_index": content_index,
        "content_source_record_id": normalize_text(content_metadata.get("source_record_id"))
        or normalize_text(content_row.get("request_id")),
        "content_payload_hash": _content_payload_hash(content_row),
    }
    if content_dataset == "raghot_qa":
        required_raghot_metadata = ("context_pack_hash", "supporting_fact_ids", "packing_policy_version")
        if any(not content_metadata.get(field) for field in required_raghot_metadata):
            raise ValueError("RAGhot QA candidate is missing required evidence provenance")
        content_provenance.update(
            {
                "context_pack_hash": content_metadata["context_pack_hash"],
                "supporting_fact_ids": content_metadata["supporting_fact_ids"],
                "packing_policy_version": content_metadata["packing_policy_version"],
            }
        )

    session_rows = [
        _sharegpt_turn(
            source_index=source_index,
            source_row=source_row,
            session_id=session_id,
            original_session_id=original_session_id,
            turn_index=turn_index,
            content_provenance=content_provenance,
        )
        for turn_index, (source_index, source_row) in enumerate(skeleton)
    ]
    followups = _followup_prompts(task)
    injected_prompts = [content_prompt, *followups]
    for turn_index, prompt in enumerate(injected_prompts, start=2):
        session_rows.append(
            {
                "request_id": f"{session_id}-turn-{turn_index}",
                "task": task,
                "prompt": prompt,
                "reference": reference,
                "session_id": session_id,
                "turn_index": turn_index,
                "metadata": _hybrid_metadata(
                    original_session_id=original_session_id,
                    **content_provenance,
                    role=("content_query", "reference_recall", "reference_rewrite")[turn_index - 2],
                    content_row=content_row,
                    content_prompt_chars=len(content_prompt),
                    last_turn=turn_index == 4,
                ),
            }
        )
    return session_rows


def _sharegpt_turn(
    *,
    source_index: int,
    source_row: dict[str, Any],
    session_id: str,
    original_session_id: str,
    turn_index: int,
    content_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": f"{session_id}-turn-{turn_index}",
        "task": normalize_text(source_row.get("task")) or "chat",
        "prompt": normalize_text(source_row.get("prompt")),
        "reference": normalize_text(source_row.get("reference")),
        "session_id": session_id,
        "turn_index": turn_index,
        "metadata": _hybrid_metadata(
            original_session_id=original_session_id,
            **content_provenance,
            role="sharegpt_opening",
            content_row=source_row,
            content_prompt_chars=len(normalize_text(source_row.get("prompt"))),
            last_turn=False,
        ),
    }


def _hybrid_metadata(
    *,
    original_session_id: str,
    content_source_dataset: str,
    content_source_request_id: str,
    content_source_index: int,
    content_source_record_id: str,
    content_payload_hash: str,
    context_pack_hash: str | None = None,
    supporting_fact_ids: list[str] | None = None,
    packing_policy_version: str | None = None,
    role: str,
    content_row: dict[str, Any],
    content_prompt_chars: int,
    last_turn: bool,
) -> dict[str, Any]:
    content_metadata = content_row.get("metadata") or {}
    metadata = {
        "source": "hybrid_session_builder",
        "source_dataset": HYBRID_SOURCE,
        "source_session_dataset": "sharegpt_longbench_multiturn_pilot",
        "content_source_dataset": content_source_dataset,
        "content_source_subdataset": normalize_text(content_metadata.get("longbench_dataset")),
        "content_source_request_id": content_source_request_id,
        "content_source_index": content_source_index,
        "content_source_record_id": content_source_record_id,
        "content_payload_hash": content_payload_hash,
        "content_prompt_chars": content_prompt_chars,
        "original_session_id": original_session_id,
        "injection_template": "template_a",
        "hybrid_turn_role": role,
        "split": "candidate",
        "risk_family": "unlabeled",
        "last_turn": last_turn,
    }
    if content_source_dataset == "raghot_qa":
        metadata.update(
            {
                "context_pack_hash": context_pack_hash,
                "supporting_fact_ids": supporting_fact_ids,
                "packing_policy_version": packing_policy_version,
            }
        )
    return metadata


def _followup_prompts(task: str) -> tuple[str, str]:
    if task == "qa":
        return (
            "Based on the LongBench question and answer from the previous turn, repeat the answer exactly without adding new facts.\nAnswer:",
            "Rewrite the previous answer concisely without adding new facts, while preserving the same answer.\nAnswer:",
        )
    if task == "summary":
        return (
            "Based on the LongBench document and summary from the previous turn, repeat its conclusion without adding new facts.\nSummary:",
            "Rewrite the previous summary concisely without adding new facts, while preserving the same conclusion.\nSummary:",
        )
    raise ValueError(f"unsupported LongBench task: {task}")


def _truncate_prompt(prompt: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_content_prompt_chars must be positive")
    return prompt[:max_chars]


def _content_payload_hash(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    known_hash = normalize_text(
        metadata.get("content_payload_hash") or metadata.get("payload_hash") or metadata.get("candidate_hash")
    )
    if known_hash:
        return known_hash
    payload = "\n".join(
        (normalize_text(row.get("task")), normalize_text(row.get("prompt")), normalize_text(row.get("reference")))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_unique_final_sources(sessions: list[list[dict[str, Any]]]) -> None:
    seen_skeletons: set[str] = set()
    seen_content: set[tuple[str, str]] = set()
    seen_payloads: set[str] = set()
    for session_rows in sessions:
        metadata = session_rows[0]["metadata"]
        original_session_id = normalize_text(metadata.get("original_session_id"))
        content_key = (
            normalize_text(metadata.get("content_source_dataset")),
            normalize_text(metadata.get("content_source_record_id")),
        )
        payload_hash = normalize_text(metadata.get("content_payload_hash"))
        if not original_session_id or original_session_id in seen_skeletons:
            raise ValueError(f"duplicate original_session_id: {original_session_id}")
        if not all(content_key) or content_key in seen_content:
            raise ValueError(f"duplicate content source record: {content_key}")
        if not payload_hash or payload_hash in seen_payloads:
            raise ValueError(f"duplicate injected payload: {payload_hash}")
        seen_skeletons.add(original_session_id)
        seen_content.add(content_key)
        seen_payloads.add(payload_hash)


def _interleave_sessions(sessions: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for turn_index in range(5):
        for session_rows in sessions:
            row = dict(session_rows[turn_index])
            row["arrival_index"] = len(exported)
            exported.append(row)
    return exported


def _session_task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if int(row["turn_index"]) != 2:
            continue
        task = str(row["task"])
        counts[task] = counts.get(task, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
EXPECTED_MAINSTREAM_PROFILES = frozenset(
    (FULL_GPU_PROFILE, *MAINSTREAM_KIVI_PROFILES, *MAINSTREAM_H2O_PROFILES)
)


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
        "rows": len(exported),
        "sessions": len({row["session_id"] for row in exported}),
        "risk_distribution": _count_by(exported, "risk_family"),
        "splits": _count_split(exported),
        "session_tasks": _count_session_tasks(exported),
        "risk_task_split_sessions": _count_risk_task_split_sessions(exported),
    }
    return exported, manifest


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
            ranked = sorted(pool, key=lambda item: (-item[4], item[2]))[:SESSIONS_PER_RISK_TASK]
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
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from build_hybrid_sessions import build_hybrid_sessions
from common import (
    RAGHOT_QA_CANDIDATES,
    ensure_artifact_dirs,
    read_jsonl,
    write_json,
    write_jsonl,
)
from label_sharegpt_sessions import read_measurements, strict_hybrid_cell_counts


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TAILGUARDKV_REPO = Path("/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline")
HYBRID_SKELETONS = WORKSPACE_ROOT / "artifacts/candidates/sharegpt_hybrid_skeletons.jsonl"
LONG_BENCH_CANDIDATES = WORKSPACE_ROOT / "artifacts/candidates/longbench_candidates.jsonl"
HYBRID_CANDIDATES = WORKSPACE_ROOT / "artifacts/candidates/hybrid_session_candidates.jsonl"
HYBRID_MEASUREMENTS = WORKSPACE_ROOT / "artifacts/measurements/hybrid_session_candidates_profiles.csv"
HYBRID_BATCH_ROOT = WORKSPACE_ROOT / "artifacts/candidates/hybrid_batches"
EXPECTED_PROFILES = (
    "full_gpu",
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
RISK_FAMILIES = ("kivi_sensitive", "h2o_sensitive", "low_risk")
SESSION_TASKS = ("qa", "summary")
REQUIRED_PER_CELL = 12
BATCH_SESSION_COUNT = 12


def select_next_hybrid_batch(
    sharegpt_rows: list[dict[str, Any]],
    longbench_rows: list[dict[str, Any]],
    *,
    raghot_rows: list[dict[str, Any]] | None,
    existing_candidate_rows: list[dict[str, Any]],
    measurements: list[Any],
    longbench_manifests: list[dict[str, Any]],
    completed_longbench_prescreens: list[dict[str, Any]],
    source_exhausted: bool,
    required_per_cell: int = REQUIRED_PER_CELL,
    batch_id: str = "hybrid_candidates_batch000",
) -> dict[str, Any]:
    """Build one final-form measurement batch without turning prescreen ranks into labels."""
    if required_per_cell <= 0:
        raise ValueError("required_per_cell must be positive")
    counts = strict_hybrid_cell_counts(existing_candidate_rows, measurements)
    missing = _missing_cells(counts, required_per_cell)
    if not missing:
        return _status_report("complete", counts, required_per_cell, missing, source_exhausted=source_exhausted)
    if source_exhausted:
        return _status_report("insufficient", counts, required_per_cell, missing, source_exhausted=True)

    qa_needed = any(str(cell["cell"]).endswith("/qa") for cell in missing)
    summary_needed = any(str(cell["cell"]).endswith("/summary") for cell in missing)
    task_counts = _batch_task_counts(qa_needed=qa_needed, summary_needed=summary_needed)
    use_raghot_qa = qa_needed and _longbench_qa_batches_complete(
        longbench_manifests, completed_longbench_prescreens
    )
    try:
        rows = build_hybrid_sessions(
            sharegpt_rows,
            longbench_rows,
            raghot_rows=raghot_rows if use_raghot_qa else None,
            use_raghot_qa=use_raghot_qa,
            session_count=BATCH_SESSION_COUNT,
            task_counts=task_counts,
            require_unique_sources=True,
        )
    except (RuntimeError, ValueError) as error:
        report = _status_report(
            "insufficient", counts, required_per_cell, missing, source_exhausted=source_exhausted
        )
        report["reason"] = str(error)
        return report

    return {
        "status": "pending_measurement",
        "batch_id": batch_id,
        "rows": rows,
        "session_count": BATCH_SESSION_COUNT,
        "request_count": len(rows),
        "expected_profiles": list(EXPECTED_PROFILES),
        "task_counts": task_counts,
        "content_qa_source": "raghot_qa" if use_raghot_qa else "longbench",
        "required_per_cell": required_per_cell,
        "strict_counts": _display_counts(counts),
        "missing_cells": missing,
        "measurement_contract": {
            "cuda_visible_devices": "0,1",
            "profile_chunk_size": 1,
            "use_persistent_workers": True,
            "serial_gpu_exclusive": True,
        },
    }


def build_async_serial_measurement_command(
    batch: dict[str, Any],
    *,
    config_path: Path,
    output_path: Path,
    log_path: Path,
    pid_path: Path,
    lock_path: Path,
) -> str:
    if batch.get("status") != "pending_measurement":
        raise ValueError("only a pending hybrid batch can be measured")
    return (
        "mkdir -p "
        f"'{output_path.parent}' '{log_path.parent}' '{pid_path.parent}' '{lock_path.parent}'; "
        f"cd '{TAILGUARDKV_REPO}' && "
        f"nohup setsid flock -n '{lock_path}' env CUDA_VISIBLE_DEVICES=0,1 "
        "conda run -n tailguardkv-base python -m run_util.build_profile_table "
        f"--config '{config_path}' --output '{output_path}' --no-dry-run "
        f"> '{log_path}' 2>&1 < /dev/null & echo $! > '{pid_path}'"
    )


def validate_hybrid_measurement_config(config_path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to validate a hybrid measurement config") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_smoke = payload.get("profile_smoke") if isinstance(payload, dict) else None
    if not isinstance(profile_smoke, dict):
        raise ValueError("persistent-worker dual-GPU measurement config requires profile_smoke")
    required = {
        "profile_chunk_size": 1,
        "use_persistent_workers": True,
        "full_cuda_visible_devices": "0,1",
        "kivi_cuda_visible_devices": "0,1",
        "h2o_cuda_visible_devices": "0,1",
    }
    invalid = [
        key
        for key, expected in required.items()
        if profile_smoke.get(key) != expected
    ]
    if invalid:
        raise ValueError(
            "persistent-worker dual-GPU measurement config is invalid: " + ",".join(invalid)
        )


def write_hybrid_batch(batch: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    if batch.get("status") != "pending_measurement":
        raise ValueError("cannot write a non-pending hybrid batch")
    batch_id = str(batch["batch_id"])
    candidate_path = output_root / f"{batch_id}.jsonl"
    manifest_path = output_root / f"{batch_id}.manifest.json"
    write_jsonl(candidate_path, list(batch["rows"]))
    manifest = {key: value for key, value in batch.items() if key != "rows"}
    manifest["candidates_jsonl"] = str(candidate_path)
    manifest["manifest_output"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def launch_hybrid_batch(
    batch: dict[str, Any],
    *,
    config_path: Path,
    output_path: Path,
    log_path: Path,
    pid_path: Path,
    lock_path: Path,
) -> None:
    validate_hybrid_measurement_config(config_path)
    command = build_async_serial_measurement_command(
        batch,
        config_path=config_path,
        output_path=output_path,
        log_path=log_path,
        pid_path=pid_path,
        lock_path=lock_path,
    )
    subprocess.run(command, shell=True, check=True)


def _missing_cells(counts: dict[tuple[str, str], int], required_per_cell: int) -> list[dict[str, Any]]:
    return [
        {"cell": f"{risk_family}/{task}", "available": counts[(risk_family, task)], "required": required_per_cell}
        for risk_family in RISK_FAMILIES
        for task in SESSION_TASKS
        if counts[(risk_family, task)] < required_per_cell
    ]


def _display_counts(counts: dict[tuple[str, str], int]) -> dict[str, int]:
    return {f"{risk_family}/{task}": counts[(risk_family, task)] for risk_family, task in counts}


def _status_report(
    status: str,
    counts: dict[tuple[str, str], int],
    required_per_cell: int,
    missing: list[dict[str, Any]],
    *,
    source_exhausted: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "required_per_cell": required_per_cell,
        "strict_counts": _display_counts(counts),
        "missing_cells": missing,
        "source_exhausted": source_exhausted,
    }


def _longbench_qa_batches_complete(
    manifests: list[dict[str, Any]], completed_prescreens: list[dict[str, Any]]
) -> bool:
    """Verify every available QA manifest has a complete local prescreen result."""
    completed_by_id = {str(item.get("batch_id", "")): item for item in completed_prescreens}
    qa_manifests = [
        manifest
        for manifest in manifests
        if any(str(candidate.get("task", "")).lower() == "qa" for candidate in manifest.get("candidates", []))
    ]
    if not qa_manifests:
        return False
    for manifest in qa_manifests:
        batch_id = str(manifest.get("batch_id", ""))
        completed = completed_by_id.get(batch_id)
        if completed is None:
            return False
        manifest_ids = {str(candidate.get("request_id", "")) for candidate in manifest.get("candidates", [])}
        completed_ids = {str(candidate.get("request_id", "")) for candidate in completed.get("candidates", [])}
        ranked_ids = {str(candidate.get("request_id", "")) for candidate in completed.get("ranked_candidates", [])}
        if not manifest_ids or manifest_ids != completed_ids or manifest_ids != ranked_ids:
            return False
    return True


def _batch_task_counts(*, qa_needed: bool, summary_needed: bool) -> dict[str, int]:
    if qa_needed and summary_needed:
        return {"qa": BATCH_SESSION_COUNT // 2, "summary": BATCH_SESSION_COUNT // 2}
    if qa_needed:
        return {"qa": BATCH_SESSION_COUNT, "summary": 0}
    return {"qa": 0, "summary": BATCH_SESSION_COUNT}


def _load_json_dir(directory: Path, pattern: str) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob(pattern))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Select one 12-session hybrid measurement batch.")
    parser.add_argument("--sharegpt", default=str(HYBRID_SKELETONS))
    parser.add_argument("--longbench", default=str(LONG_BENCH_CANDIDATES))
    parser.add_argument("--raghot", default=str(RAGHOT_QA_CANDIDATES))
    parser.add_argument("--existing-candidates", default=str(HYBRID_CANDIDATES))
    parser.add_argument("--measurements", default=str(HYBRID_MEASUREMENTS))
    parser.add_argument("--longbench-manifests-dir", required=True)
    parser.add_argument("--completed-prescreens-dir", required=True)
    parser.add_argument("--output-root", default=str(HYBRID_BATCH_ROOT))
    parser.add_argument("--batch-id", default="hybrid_candidates_batch000")
    parser.add_argument("--source-exhausted", action="store_true")
    parser.add_argument("--required-per-cell", type=int, default=REQUIRED_PER_CELL)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()

    batch = select_next_hybrid_batch(
        read_jsonl(Path(args.sharegpt)),
        read_jsonl(Path(args.longbench)),
        raghot_rows=read_jsonl(Path(args.raghot)),
        existing_candidate_rows=read_jsonl(Path(args.existing_candidates)),
        measurements=read_measurements(Path(args.measurements)),
        longbench_manifests=_load_json_dir(Path(args.longbench_manifests_dir), "*.manifest.json"),
        completed_longbench_prescreens=_load_json_dir(Path(args.completed_prescreens_dir), "*.prescreen.json"),
        source_exhausted=args.source_exhausted,
        required_per_cell=args.required_per_cell,
        batch_id=args.batch_id,
    )
    if batch["status"] != "pending_measurement":
        print(json.dumps(batch, ensure_ascii=False))
        return 1 if batch["status"] == "insufficient" else 0
    ensure_artifact_dirs()
    manifest = write_hybrid_batch(batch, output_root=Path(args.output_root))
    if args.launch:
        if not args.config:
            raise SystemExit("--launch requires --config")
        root = Path(args.output_root)
        launch_hybrid_batch(
            batch,
            config_path=Path(args.config),
            output_path=root / f"{args.batch_id}.csv",
            log_path=root / "logs" / f"{args.batch_id}.log",
            pid_path=root / "pids" / f"{args.batch_id}.pid",
            lock_path=root / "locks" / "hybrid_gpu.lock",
        )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_hybrid_sessions import build_hybrid_sessions
from label_sharegpt_sessions import (
    build_fixture,
    classify_profile_losses,
    select_hybrid_candidate_reserves,
)
from select_hybrid_candidate_batches import (
    build_async_serial_measurement_command,
    select_next_hybrid_batch,
    validate_hybrid_measurement_config,
)


KIVI_PROFILE = "kivi_4bit_residual32"
H2O_PROFILE = "h2o_heavy10_recent10"
KIVI_PROFILES = (
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
)
H2O_PROFILES = (
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
MAINSTREAM_PROFILES = ("full_gpu", *KIVI_PROFILES, *H2O_PROFILES)


def _sharegpt_rows(count: int = 48) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session_index in range(count):
        source_session_id = f"source-session-{session_index:03d}"
        for turn_index in range(2):
            rows.append(
                {
                    "request_id": f"sharegpt-{session_index:03d}-{turn_index}",
                    "task": "chat",
                    "prompt": f"sharegpt prompt {session_index}/{turn_index}",
                    "reference": f"sharegpt answer {session_index}/{turn_index}",
                    "session_id": source_session_id,
                    "turn_index": turn_index,
                    "metadata": {"source_dataset": "sharegpt_pilot"},
                }
            )
    return rows


def _longbench_rows(count: int = 48) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        task = "qa" if index % 2 == 0 else "summary"
        rows.append(
            {
                "request_id": f"longbench-{task}-{index:03d}",
                "task": task,
                "prompt": f"longbench {task} prompt {index}",
                "reference": f"longbench {task} answer {index}",
                "metadata": {
                    "source_dataset": f"longbench_{task}_dataset",
                    "longbench_dataset": f"{task}_dataset",
                },
            }
        )
    return rows


def _raghot_rows(count: int = 24) -> list[dict[str, object]]:
    return [
        {
            "request_id": f"raghot_qa_{index:03d}",
            "task": "qa",
            "prompt": f"raghot question {index}",
            "reference": f"raghot answer {index}",
            "metadata": {
                "source_dataset": "raghot_qa",
                "source_record_id": f"raghot-source-{index:03d}",
                "context_pack_hash": f"context-{index:03d}",
                "supporting_fact_ids": [f"fact-{index:03d}"],
                "packing_policy_version": "raghot_support_first_v1",
                "content_payload_hash": f"payload-{index:03d}",
            },
        }
        for index in range(count)
    ]


def _by_session(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    return grouped


def test_builds_48_complete_five_turn_sessions_with_hybrid_provenance() -> None:
    rows = build_hybrid_sessions(_sharegpt_rows(), _longbench_rows())

    sessions = _by_session(rows)
    assert len(rows) == 240
    assert len(sessions) == 48
    assert {tuple(int(row["turn_index"]) for row in session_rows) for session_rows in sessions.values()} == {
        (0, 1, 2, 3, 4)
    }

    first_session = sessions["hybrid-session-000"]
    assert [row["prompt"] for row in first_session[:2]] == [
        "sharegpt prompt 0/0",
        "sharegpt prompt 0/1",
    ]
    for row in first_session:
        metadata = row["metadata"]
        assert metadata["source"] == "hybrid_session_builder"
        assert metadata["source_dataset"] == "sharegpt_longbench_hybrid_session"
        assert metadata["original_session_id"] == "source-session-000"
        assert metadata["injection_template"]
        assert metadata["content_source_dataset"]
        assert metadata["content_source_request_id"]
        assert isinstance(metadata["content_source_index"], int)


def test_injected_turns_keep_longbench_task_and_repeat_its_reference() -> None:
    rows = build_hybrid_sessions(_sharegpt_rows(), _longbench_rows())
    sessions = _by_session(rows)

    qa_turns = sessions["hybrid-session-000"][2:]
    summary_turns = sessions["hybrid-session-001"][2:]
    assert [row["task"] for row in qa_turns] == ["qa", "qa", "qa"]
    assert [row["reference"] for row in qa_turns] == ["longbench qa answer 0"] * 3
    assert "repeat" in str(qa_turns[1]["prompt"]).lower()
    assert "rewrite" in str(qa_turns[2]["prompt"]).lower()

    assert [row["task"] for row in summary_turns] == ["summary", "summary", "summary"]
    assert [row["reference"] for row in summary_turns] == ["longbench summary answer 1"] * 3
    assert "repeat" in str(summary_turns[1]["prompt"]).lower()
    assert "rewrite" in str(summary_turns[2]["prompt"]).lower()

    for turn in (*qa_turns, *summary_turns):
        metadata = turn["metadata"]
        assert metadata["content_source_dataset"] == "longbench"
        assert metadata["content_source_request_id"].startswith("longbench-")


def test_candidate_arrivals_are_globally_interleaved_by_turn() -> None:
    rows = build_hybrid_sessions(_sharegpt_rows(), _longbench_rows())

    assert [row["arrival_index"] for row in rows] == list(range(240))
    assert [row["turn_index"] for row in rows[:48]] == [0] * 48
    assert [row["turn_index"] for row in rows[48:96]] == [1] * 48
    assert len({row["session_id"] for row in rows[:48]}) == 48


def test_injected_longbench_prompt_is_truncated_to_the_configured_limit() -> None:
    longbench_rows = _longbench_rows()
    longbench_rows[0]["prompt"] = "x" * 4096

    rows = build_hybrid_sessions(_sharegpt_rows(), longbench_rows, max_content_prompt_chars=512)

    injected_turn = next(row for row in rows if row["turn_index"] == 2)
    assert len(str(injected_turn["prompt"])) == 512


def test_raghot_builds_qa_only_with_session_consistent_content_provenance() -> None:
    rows = build_hybrid_sessions(
        _sharegpt_rows(12),
        _longbench_rows(12),
        raghot_rows=_raghot_rows(12),
        use_raghot_qa=True,
        session_count=12,
    )

    for session_rows in _by_session(rows).values():
        assert {row["task"] for row in session_rows[2:]} in ({"qa"}, {"summary"})
        metadata = [row["metadata"] for row in session_rows]
        assert len({item["original_session_id"] for item in metadata}) == 1
        assert len({item["content_source_dataset"] for item in metadata}) == 1
        assert len({item["content_source_request_id"] for item in metadata}) == 1
        assert len({item["content_payload_hash"] for item in metadata}) == 1
        assert [item["hybrid_turn_role"] for item in metadata] == [
            "sharegpt_opening",
            "sharegpt_opening",
            "content_query",
            "reference_recall",
            "reference_rewrite",
        ]
        if metadata[0]["content_source_dataset"] == "raghot_qa":
            assert metadata[0]["context_pack_hash"].startswith("context-")
            assert metadata[0]["supporting_fact_ids"]
            assert metadata[0]["packing_policy_version"] == "raghot_support_first_v1"


def test_raghot_summary_content_is_rejected() -> None:
    raghots = _raghot_rows(6)
    raghots[0]["task"] = "summary"

    with pytest.raises(ValueError, match="RAGhot.*QA"):
        build_hybrid_sessions(
            _sharegpt_rows(12),
            _longbench_rows(12),
            raghot_rows=raghots,
            use_raghot_qa=True,
            session_count=12,
        )


def _hybrid_rows() -> list[dict[str, object]]:
    return build_hybrid_sessions(_sharegpt_rows(), _longbench_rows())


def _fixture_rows() -> list[dict[str, object]]:
    return build_hybrid_sessions(_sharegpt_rows(72), _longbench_rows(72), session_count=72)


def _longbench_completion_manifests(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest = {"batch_id": "longbench_prescreen_batch000", "candidates": rows}
    completed = {
        "batch_id": manifest["batch_id"],
        "candidates": rows,
        "ranked_candidates": [{"request_id": row["request_id"]} for row in rows],
    }
    return [manifest], [completed]


def _balanced_measurements(rows: list[dict[str, object]]) -> list[SimpleNamespace]:
    session_ids_by_task: dict[str, list[str]] = defaultdict(list)
    later_rows_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["turn_index"] == 2:
            session_ids_by_task[str(row["task"])].append(str(row["session_id"]))
        if int(row["turn_index"]) >= 2:
            later_rows_by_session[str(row["session_id"])].append(row)

    losses_by_risk = {
        "kivi_sensitive": {
            "full_gpu": 0.0,
            **dict.fromkeys(KIVI_PROFILES, 0.07),
            **dict.fromkeys(H2O_PROFILES, 0.02),
        },
        "h2o_sensitive": {
            "full_gpu": 0.0,
            **dict.fromkeys(KIVI_PROFILES, 0.02),
            **dict.fromkeys(H2O_PROFILES, 0.07),
        },
        "low_risk": dict.fromkeys(MAINSTREAM_PROFILES, 0.005),
    }
    measurements: list[SimpleNamespace] = []
    for task in ("qa", "summary"):
        per_risk = 12 if len(session_ids_by_task[task]) >= 36 else 8
        for index, session_id in enumerate(session_ids_by_task[task]):
            risk_family = ("kivi_sensitive", "h2o_sensitive", "low_risk")[index // per_risk]
            for candidate_row in later_rows_by_session[session_id]:
                for profile, loss in losses_by_risk[risk_family].items():
                    measurements.append(
                        SimpleNamespace(
                            request_id=candidate_row["request_id"],
                            session_id=session_id,
                            turn_index=candidate_row["turn_index"],
                            profile=profile,
                            quality_loss=loss,
                            ok=True,
                            measured=True,
                        )
                    )
    return measurements


def test_strict_risk_thresholds_require_cross_family_gap_and_exclude_ties() -> None:
    def losses(kivi: float, h2o: float) -> dict[str, float]:
        return {
            "full_gpu": 0.0,
            **dict.fromkeys(KIVI_PROFILES, kivi),
            **dict.fromkeys(H2O_PROFILES, h2o),
        }

    assert classify_profile_losses(losses(0.05, 0.03)) == "kivi_sensitive"
    assert classify_profile_losses(losses(0.03, 0.05)) == "h2o_sensitive"
    assert classify_profile_losses(losses(0.01, 0.01)) == "low_risk"
    assert classify_profile_losses(losses(0.06, 0.04001)) is None
    assert classify_profile_losses(losses(0.06, 0.05)) is None
    assert classify_profile_losses({KIVI_PROFILE: 0.07, H2O_PROFILE: 0.01}) is None


def test_fixture_selects_four_calibration_and_four_eval_sessions_per_risk_and_task() -> None:
    candidate_rows = _fixture_rows()
    fixture, manifest = build_fixture(candidate_rows, _balanced_measurements(candidate_rows))

    turn_two_rows = [row for row in fixture if row["turn_index"] == 2]
    session_counts = Counter(
        (row["metadata"]["risk_family"], row["task"], row["metadata"]["split"])
        for row in turn_two_rows
    )
    assert len(fixture) == 240
    assert manifest["rows"] == 240
    assert manifest["sessions"] == 48
    assert set(session_counts.values()) == {4}
    assert set(session_counts) == {
        (risk_family, task, split)
        for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk")
        for task in ("qa", "summary")
        for split in ("calibration", "eval")
    }

    assert [row["arrival_index"] for row in fixture] == list(range(240))
    assert [row["turn_index"] for row in fixture[:48]] == [0] * 48
    for session_rows in _by_session(fixture).values():
        assert len({row["metadata"]["risk_family"] for row in session_rows}) == 1
        assert len({row["metadata"]["split"] for row in session_rows}) == 1


def test_fixture_raises_when_any_strict_risk_task_pool_is_insufficient() -> None:
    candidate_rows = _fixture_rows()
    measurements = _balanced_measurements(candidate_rows)
    ambiguous_session = next(
        str(row["session_id"])
        for row in candidate_rows
        if row["turn_index"] == 2 and row["task"] == "qa"
    )
    measurements = [
        SimpleNamespace(
            **{
                **vars(row),
                "quality_loss": 0.06 if row.profile in KIVI_PROFILES else 0.05,
            }
        )
        if row.session_id == ambiguous_session
        else row
        for row in measurements
    ]

    try:
        build_fixture(candidate_rows, measurements)
    except RuntimeError as error:
        assert "kivi_sensitive" in str(error)
        assert "qa" in str(error)
        assert "available=11 required=12" in str(error)
    else:
        raise AssertionError("expected an insufficient strict risk/task pool error")


def test_fixture_rejects_measurement_with_foreign_request_id() -> None:
    candidate_rows = _hybrid_rows()
    measurements = _balanced_measurements(candidate_rows)
    measurements.append(
        SimpleNamespace(
            request_id="foreign-request",
            session_id="hybrid-session-000",
            turn_index=2,
            profile=KIVI_PROFILE,
            quality_loss=0.9,
            ok=True,
            measured=True,
        )
    )

    with pytest.raises(ValueError, match="foreign-request"):
        build_fixture(candidate_rows, measurements)


def test_fixture_rejects_non_hybrid_candidate_provenance() -> None:
    candidate_rows = _hybrid_rows()
    candidate_rows[0]["metadata"] = {
        **candidate_rows[0]["metadata"],
        "source": "legacy_sharegpt_builder",
    }

    with pytest.raises(ValueError, match="hybrid provenance"):
        build_fixture(candidate_rows, _balanced_measurements(candidate_rows))


def test_fixture_rejects_missing_or_failed_mainstream_profile() -> None:
    candidate_rows = _hybrid_rows()
    measurements = _balanced_measurements(candidate_rows)
    target = next(
        row
        for row in measurements
        if row.session_id == "hybrid-session-000" and row.turn_index == 2 and row.profile == KIVI_PROFILE
    )

    missing = [row for row in measurements if row is not target]
    with pytest.raises(RuntimeError, match="missing_profiles"):
        build_fixture(candidate_rows, missing)

    failed = [
        SimpleNamespace(**{**vars(row), "ok": False}) if row is target else row
        for row in measurements
    ]
    with pytest.raises(RuntimeError, match="failed_profiles"):
        build_fixture(candidate_rows, failed)


def test_fixture_rejects_incomplete_turn_2_to_4_measurement_coverage() -> None:
    candidate_rows = _hybrid_rows()
    measurements = [
        row
        for row in _balanced_measurements(candidate_rows)
        if not (row.session_id == "hybrid-session-000" and row.turn_index == 4)
    ]

    with pytest.raises(RuntimeError, match="turn=4"):
        build_fixture(candidate_rows, measurements)


def test_hybrid_reserves_require_twelve_complete_unique_candidates_per_cell() -> None:
    rows = _hybrid_rows()
    measurements = _balanced_measurements(rows)

    with pytest.raises(RuntimeError, match=r"kivi_sensitive/qa available=8 required=12 source_exhausted=True"):
        select_hybrid_candidate_reserves(rows, measurements, required_per_cell=12, source_exhausted=True)


def test_hybrid_reserves_reject_duplicate_skeleton_content_and_incomplete_profiles() -> None:
    rows = _hybrid_rows()
    measurements = _balanced_measurements(rows)
    duplicate_skeleton = [dict(row) for row in rows]
    for row in duplicate_skeleton:
        if row["session_id"] == "hybrid-session-001":
            row["metadata"] = {
                **row["metadata"],
                "original_session_id": "source-session-000",
            }
    with pytest.raises(ValueError, match="duplicate original_session_id"):
        select_hybrid_candidate_reserves(duplicate_skeleton, measurements, required_per_cell=8)

    duplicate_content = [dict(row) for row in rows]
    for row in duplicate_content:
        if row["session_id"] == "hybrid-session-001":
            row["metadata"] = {
                **row["metadata"],
                "content_source_record_id": "longbench-qa-000",
            }
    with pytest.raises(ValueError, match="duplicate content source record"):
        select_hybrid_candidate_reserves(duplicate_content, measurements, required_per_cell=8)

    incomplete = [
        row
        for row in measurements
        if not (row.session_id == "hybrid-session-000" and row.profile == "full_gpu")
    ]
    with pytest.raises(RuntimeError, match="missing_profiles"):
        select_hybrid_candidate_reserves(rows, incomplete, required_per_cell=8)


def test_next_hybrid_batch_is_twelve_sessions_sixty_requests_and_serial_async() -> None:
    longbench_rows = _longbench_rows(12)
    manifests, completed = _longbench_completion_manifests(longbench_rows)
    batch = select_next_hybrid_batch(
        _sharegpt_rows(12),
        longbench_rows,
        raghot_rows=_raghot_rows(12),
        existing_candidate_rows=[],
        measurements=[],
        longbench_manifests=manifests,
        completed_longbench_prescreens=completed,
        source_exhausted=False,
    )

    assert len(batch["rows"]) == 60
    assert batch["session_count"] == 12
    assert batch["request_count"] == 60
    command = build_async_serial_measurement_command(
        batch,
        config_path=Path("configs/hybrid.yaml"),
        output_path=Path("/tmp/hybrid.csv"),
        log_path=Path("/tmp/hybrid.log"),
        pid_path=Path("/tmp/hybrid.pid"),
        lock_path=Path("/tmp/hybrid.lock"),
    )
    assert "nohup setsid flock -n" in command
    assert "CUDA_VISIBLE_DEVICES=0,1" in command
    assert "--no-dry-run" in command


def test_final_export_requires_twelve_complete_candidates_per_risk_task_cell() -> None:
    rows = _hybrid_rows()
    with pytest.raises(RuntimeError, match=r"kivi_sensitive/qa available=8 required=12 source_exhausted=False"):
        build_fixture(rows, _balanced_measurements(rows))


def test_final_export_rejects_inconsistent_five_turn_roles_and_provenance() -> None:
    rows = _fixture_rows()
    target = next(row for row in rows if row["session_id"] == "hybrid-session-000" and row["turn_index"] == 2)
    target["metadata"] = {**target["metadata"], "hybrid_turn_role": "reference_recall"}

    with pytest.raises(ValueError, match="invalid hybrid session roles"):
        build_fixture(rows, _balanced_measurements(rows))


def test_raghot_fallback_requires_longbench_exhaustion_report_and_underfilled_qa() -> None:
    longbench_rows = _longbench_rows(12)
    longbench_rows = [row for row in longbench_rows if row["task"] == "summary"] + [
        row for row in longbench_rows if row["task"] == "qa"
    ][:3]
    manifests, completed = _longbench_completion_manifests(longbench_rows)
    unverified = select_next_hybrid_batch(
        _sharegpt_rows(12),
        longbench_rows,
        raghot_rows=_raghot_rows(12),
        existing_candidate_rows=[],
        measurements=[],
        longbench_manifests=manifests,
        completed_longbench_prescreens=[],
        source_exhausted=False,
    )
    assert unverified["status"] == "insufficient"
    assert unverified["source_exhausted"] is False

    verified = select_next_hybrid_batch(
        _sharegpt_rows(12),
        longbench_rows,
        raghot_rows=_raghot_rows(12),
        existing_candidate_rows=[],
        measurements=[],
        longbench_manifests=manifests,
        completed_longbench_prescreens=completed,
        source_exhausted=False,
    )
    assert verified["status"] == "pending_measurement"
    assert verified["content_qa_source"] == "raghot_qa"


def test_batch_construction_failure_preserves_actual_source_exhaustion_state() -> None:
    longbench_rows = _longbench_rows(2)
    batch = select_next_hybrid_batch(
        _sharegpt_rows(12),
        longbench_rows,
        raghot_rows=_raghot_rows(12),
        existing_candidate_rows=[],
        measurements=[],
        longbench_manifests=[],
        completed_longbench_prescreens=[],
        source_exhausted=False,
    )
    assert batch["status"] == "insufficient"
    assert batch["source_exhausted"] is False


def test_final_export_requires_content_turns_to_share_task_and_reference() -> None:
    rows = _fixture_rows()
    target = next(row for row in rows if row["session_id"] == "hybrid-session-000" and row["turn_index"] == 4)
    target["reference"] = "mismatched reference"

    with pytest.raises(ValueError, match="inconsistent hybrid content reference"):
        build_fixture(rows, _balanced_measurements(rows))

    rows = _fixture_rows()
    target = next(row for row in rows if row["session_id"] == "hybrid-session-000" and row["turn_index"] == 4)
    target["task"] = "summary"
    with pytest.raises(ValueError, match="inconsistent hybrid content task"):
        build_fixture(rows, _balanced_measurements(rows))


def test_final_export_rejects_raghot_summary_content() -> None:
    rows = build_hybrid_sessions(
        _sharegpt_rows(72),
        _longbench_rows(72),
        raghot_rows=_raghot_rows(72),
        use_raghot_qa=True,
        session_count=72,
    )
    measurements = _balanced_measurements(rows)
    for row in rows:
        if row["session_id"] == "hybrid-session-000" and row["turn_index"] >= 2:
            row["task"] = "summary"

    with pytest.raises(ValueError, match="RAGhot Summary"):
        build_fixture(rows, measurements)


def test_hybrid_measurement_launcher_requires_persistent_dual_gpu_config(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "profile_smoke:\n"
        "  profile_chunk_size: 1\n"
        "  use_persistent_workers: true\n"
        "  full_cuda_visible_devices: '0,1'\n"
        "  kivi_cuda_visible_devices: '0,1'\n"
        "  h2o_cuda_visible_devices: '0,1'\n",
        encoding="utf-8",
    )
    validate_hybrid_measurement_config(valid)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("profile_smoke:\n  use_persistent_workers: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="persistent-worker dual-GPU"):
        validate_hybrid_measurement_config(invalid)
