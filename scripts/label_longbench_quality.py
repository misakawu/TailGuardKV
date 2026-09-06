#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    BASELINE_QUALITY_FIXTURE,
    BASELINE_QUALITY_MANIFEST,
    LOW_RISK_THRESHOLD,
    LONG_BENCH_CANDIDATES,
    LONG_BENCH_MEASUREMENTS,
    MAINSTREAM_H2O_PROFILES,
    MAINSTREAM_KIVI_PROFILES,
    SENSITIVE_THRESHOLD,
    TIE_EPSILON,
    ensure_artifact_dirs,
    ensure_repo_import_path,
    read_jsonl,
    write_json,
    write_jsonl,
)

ensure_repo_import_path()

from run_util.io_utils import read_measurements


TARGETS = {
    "kivi_sensitive": 60,
    "h2o_sensitive": 60,
    "low_risk": 60,
}
FULL_GPU_PROFILE = "full_gpu"
EXPECTED_PROFILES = (FULL_GPU_PROFILE, *MAINSTREAM_KIVI_PROFILES, *MAINSTREAM_H2O_PROFILES)
FAMILY_GAP = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description="Label LongBench measurements and export baseline_quality fixture.")
    parser.add_argument("--candidates", default=str(LONG_BENCH_CANDIDATES))
    parser.add_argument("--measurements", default=str(LONG_BENCH_MEASUREMENTS))
    parser.add_argument("--output", default=str(BASELINE_QUALITY_FIXTURE))
    parser.add_argument("--manifest", default=str(BASELINE_QUALITY_MANIFEST))
    args = parser.parse_args()

    ensure_artifact_dirs()
    candidates = {row["request_id"]: row for row in read_jsonl(Path(args.candidates))}
    measurements = read_measurements(Path(args.measurements))
    fixture, manifest = build_fixture(candidates, measurements)
    write_jsonl(Path(args.output), fixture)
    write_json(Path(args.manifest), manifest)
    print(json.dumps({"output": args.output, "rows": len(fixture)}, ensure_ascii=False))
    return 0


def build_fixture(candidates: dict[str, dict[str, Any]], measurements: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise RuntimeError(
            "direct prescreen measurements cannot create a final quality fixture; "
            "fully shaped final-input candidates are required"
        )
    validated = _validate_final_quality_inputs(candidates, measurements)
    strict_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_ties: list[str] = []
    for item in validated:
        risk_family = classify_profile_losses(item["profile_losses"])
        if risk_family == "tie_sensitive":
            rejected_ties.append(str(item["candidate"]["request_id"]))
            continue
        if risk_family is None:
            continue
        item["risk_family"] = risk_family
        item["risk_margin"] = _risk_margin(risk_family, item["profile_losses"])
        strict_rows[risk_family].append(item)

    selected_items: list[dict[str, Any]] = []
    for risk_family in TARGETS:
        primary = _rank_quality_rows(
            [item for item in strict_rows[risk_family] if _source_kind(item["candidate"]) == "longbench"]
        )
        target = TARGETS[risk_family]
        selected = primary[:target]
        if len(selected) < target:
            fallback = _rank_quality_rows(
                [item for item in strict_rows[risk_family] if _source_kind(item["candidate"]) == "raghot_qa"]
            )
            if any(str(item["candidate"].get("task", "")).lower() != "qa" for item in fallback):
                fallback = []
            selected.extend(fallback[: target - len(selected)])
        if len(selected) < target:
            raise RuntimeError(
                f"{risk_family} strict final-form candidates insufficient: {len(selected)} < {target}"
            )
        selected_items.extend(selected)

    fixture: list[dict[str, Any]] = []
    for risk_family in TARGETS:
        family_rows = _rank_quality_rows(
            [item for item in selected_items if item["risk_family"] == risk_family]
        )
        for index, item in enumerate(family_rows):
            candidate = item["candidate"]
            metadata = dict(candidate["metadata"])
            metadata.update(
                {
                    "risk_family": risk_family,
                    "split": "calibration" if index < TARGETS[risk_family] // 2 else "eval",
                    "source_hash": item["source_hash"],
                    "kivi_max_loss": round(_family_max(item["profile_losses"], MAINSTREAM_KIVI_PROFILES), 6),
                    "h2o_max_loss": round(_family_max(item["profile_losses"], MAINSTREAM_H2O_PROFILES), 6),
                }
            )
            fixture.append({**candidate, "metadata": metadata})

    fixture.sort(key=lambda row: (str(row["metadata"]["risk_family"]), str(row["request_id"])))
    manifest = {
        "schema_version": 1,
        "fixture_hash": _fixture_hash(fixture),
        "rows": len(fixture),
        "risk_distribution": _count_by(fixture, "risk_family"),
        "task_distribution": _count_task(fixture),
        "source_distribution": _count_source(fixture),
        "splits": _count_split(fixture),
        "source_candidates": len(candidates),
        "strict_risk_distribution": {
            risk_family: len(strict_rows[risk_family]) for risk_family in TARGETS
        },
        "rejected_ties": sorted(rejected_ties),
        "complete_profile_coverage": {
            "profiles": list(EXPECTED_PROFILES),
            "rows_per_profile": {profile: len(fixture) for profile in EXPECTED_PROFILES},
        },
    }
    return fixture, manifest


def _fixture_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_profile_losses(profile_losses: dict[str, float]) -> str | None:
    if not set(EXPECTED_PROFILES).issubset(profile_losses):
        return None
    kivi_max = _family_max(profile_losses, MAINSTREAM_KIVI_PROFILES)
    h2o_max = _family_max(profile_losses, MAINSTREAM_H2O_PROFILES)
    overall_max = max(kivi_max, h2o_max)
    if overall_max <= LOW_RISK_THRESHOLD:
        return "low_risk"
    kivi_sensitive = kivi_max >= SENSITIVE_THRESHOLD
    h2o_sensitive = h2o_max >= SENSITIVE_THRESHOLD
    if kivi_sensitive and kivi_max - h2o_max >= FAMILY_GAP:
        return "kivi_sensitive"
    if h2o_sensitive and h2o_max - kivi_max >= FAMILY_GAP:
        return "h2o_sensitive"
    if kivi_sensitive or h2o_sensitive or abs(kivi_max - h2o_max) <= TIE_EPSILON:
        return "tie_sensitive"
    return None


def _validate_final_quality_inputs(
    candidates: dict[str, dict[str, Any]], measurements: list[Any]
) -> list[dict[str, Any]]:
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for request_id, candidate in candidates.items():
        if str(candidate.get("request_id", "")) != str(request_id):
            raise ValueError(f"candidate request_id mismatch: {request_id}")
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"candidate metadata missing: {request_id}")
        for field in ("source", "source_dataset"):
            if not str(metadata.get(field, "")).strip():
                raise ValueError(f"candidate provenance missing {field}: {request_id}")
        if not str(metadata.get("source_id") or metadata.get("source_record_id") or "").strip():
            raise ValueError(f"candidate provenance missing source ID: {request_id}")
        if _source_kind(candidate) not in {"longbench", "raghot_qa"}:
            raise ValueError(f"unsupported final quality source: {request_id}")
        source_hash = str(metadata.get("candidate_hash") or _candidate_hash(candidate))
        if metadata.get("candidate_hash") and _candidate_hash(candidate) != source_hash and _source_kind(candidate) == "longbench":
            raise ValueError(f"candidate_hash mismatch: {request_id}")
        candidate_by_id[str(request_id)] = candidate

    profiles_by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for measurement in measurements:
        request_id = str(getattr(measurement, "request_id", ""))
        profile = str(getattr(measurement, "profile", ""))
        if profile not in EXPECTED_PROFILES:
            continue
        if request_id not in candidate_by_id:
            raise ValueError(f"measurement source request is not a final candidate: {request_id}")
        if profile in profiles_by_id[request_id]:
            raise ValueError(f"duplicate final measurement: {request_id}/{profile}")
        profiles_by_id[request_id][profile] = measurement

    validated: list[dict[str, Any]] = []
    for request_id, candidate in candidate_by_id.items():
        profile_rows = profiles_by_id[request_id]
        missing = sorted(set(EXPECTED_PROFILES) - set(profile_rows))
        if missing:
            raise RuntimeError(f"final-form measurement coverage incomplete: {request_id} missing_profiles={missing}")
        source_hash = str(candidate["metadata"].get("candidate_hash") or _candidate_hash(candidate))
        profile_losses: dict[str, float] = {}
        for profile in EXPECTED_PROFILES:
            measurement = profile_rows[profile]
            if (
                not bool(getattr(measurement, "ok", False))
                or not bool(getattr(measurement, "measured", False))
                or getattr(measurement, "quality_loss", None) is None
            ):
                raise RuntimeError(f"final-form measurement invalid: {request_id}/{profile}")
            measured_hash = _measurement_source_hash(measurement, candidate)
            if measured_hash != source_hash:
                raise ValueError(f"measurement source hash mismatch: {request_id}/{profile}")
            profile_losses[profile] = float(getattr(measurement, "quality_loss"))
        validated.append({"candidate": candidate, "profile_losses": profile_losses, "source_hash": source_hash})
    return validated


def _measurement_source_hash(measurement: Any, candidate: dict[str, Any]) -> str:
    extra = getattr(measurement, "extra", {}) or {}
    direct = str(extra.get("candidate_hash") or extra.get("source_hash") or "").strip()
    if direct:
        return direct
    prompt = str(extra.get("prompt_text") or "")
    reference = str(extra.get("reference") or "")
    task = str(extra.get("task") or "")
    if prompt and reference and task:
        payload = f"{getattr(measurement, 'request_id', '')}\n{task}\n{prompt}\n{reference}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return ""


def _candidate_hash(candidate: dict[str, Any]) -> str:
    payload = f"{candidate['request_id']}\n{candidate['task']}\n{candidate['prompt']}\n{candidate['reference']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_kind(candidate: dict[str, Any]) -> str:
    source_dataset = str((candidate.get("metadata") or {}).get("source_dataset", "")).lower()
    if source_dataset.startswith("longbench"):
        return "longbench"
    if source_dataset == "raghot_qa":
        return "raghot_qa"
    return "unknown"


def _risk_margin(risk_family: str, profile_losses: dict[str, float]) -> float:
    kivi_max = _family_max(profile_losses, MAINSTREAM_KIVI_PROFILES)
    h2o_max = _family_max(profile_losses, MAINSTREAM_H2O_PROFILES)
    if risk_family == "kivi_sensitive":
        return kivi_max - h2o_max
    if risk_family == "h2o_sensitive":
        return h2o_max - kivi_max
    return -max(kivi_max, h2o_max)


def _rank_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            -float(item["risk_margin"]),
            int((item["candidate"].get("metadata") or {}).get("candidate_order", 0)),
            str(item["candidate"]["request_id"]),
        ),
    )


def _family_max(profile_losses: dict[str, float], profiles: tuple[str, ...]) -> float:
    return max([profile_losses.get(profile, 0.0) for profile in profiles] or [0.0])


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_task(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get("task", ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_source(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get("source_dataset", ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get("split", ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
