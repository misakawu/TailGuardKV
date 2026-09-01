#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from run_util.core_types import ProfileMeasurement
from run_util.io_utils import read_measurements
from scripts.generate_pilot_session_trace_requests import MAINSTREAM_H2O_PROFILES, MAINSTREAM_KIVI_PROFILES


QUALITY_GATE_THRESHOLD = 0.02


@dataclass(frozen=True)
class TraceQualityValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    qualifying_profiles: list[str] = field(default_factory=list)
    covered_tasks: set[str] = field(default_factory=set)
    profile_means: dict[str, float] = field(default_factory=dict)
    group_means: dict[str, dict[str, float]] = field(default_factory=dict)
    task_coverage: dict[str, set[str]] = field(default_factory=dict)
    sensitive_control_gaps: dict[str, float | None] = field(default_factory=dict)
    provenance_failures: list[str] = field(default_factory=list)
    quality_records: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["covered_tasks"] = sorted(self.covered_tasks)
        payload["task_coverage"] = {
            risk_family: sorted(tasks)
            for risk_family, tasks in sorted(self.task_coverage.items())
        }
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pilot session trace quality coverage before policy sweep.")
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    measurements = read_measurements(Path(args.measurements))
    fixture_rows = _load_fixture_rows(Path(args.requests))
    result = validate_trace_quality(measurements, fixture_rows)
    payload = result.to_json()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if result.passed else 2


def validate_trace_quality(
    measurements: list[ProfileMeasurement],
    fixture_rows: list[dict[str, Any]],
) -> TraceQualityValidationResult:
    fixture_by_id: dict[str, dict[str, Any]] = {}
    provenance_failures: list[str] = []
    duplicate_fixture_ids: set[str] = set()
    for fixture_row in fixture_rows:
        request_id = str(fixture_row.get("request_id", ""))
        if not request_id:
            continue
        if request_id in fixture_by_id:
            duplicate_fixture_ids.add(request_id)
            continue
        fixture_by_id[request_id] = fixture_row

    mainstream_profiles = set(MAINSTREAM_KIVI_PROFILES) | set(MAINSTREAM_H2O_PROFILES)
    quality_records: list[dict[str, Any]] = []
    eval_measurements: list[ProfileMeasurement] = []
    for measurement in measurements:
        if not measurement.ok or not measurement.measured or measurement.quality_loss is None:
            continue
        if measurement.profile not in mainstream_profiles:
            continue

        request_id = _fixture_request_id(measurement)
        fixture_row = fixture_by_id.get(request_id)
        if fixture_row is None:
            if str(measurement.extra.get("split", "")) == "eval":
                provenance_failures.append(
                    "eval quality record 无法关联 fixture: "
                    f"measurement_request_id={measurement.request_id} fixture_request_id={request_id}"
                )
            continue

        metadata = fixture_row.get("metadata") or {}
        if str(metadata.get("split", "")) != "eval":
            continue
        task = str(fixture_row.get("task", "")).strip().lower()
        if task not in {"qa", "summary"}:
            continue
        if request_id in duplicate_fixture_ids:
            provenance_failures.append(f"fixture request_id 重复，eval 质量记录无法唯一追溯: request_id={request_id}")
            continue

        failure = _quality_record_provenance_failure(fixture_row)
        measurement_split = str(measurement.extra.get("split", ""))
        if failure is None and measurement_split not in {"", "eval"}:
            failure = f"measurement split 与 eval fixture 不一致: split={measurement_split}"
        measurement_task = str(measurement.extra.get("task", "")).strip().lower()
        if failure is None and measurement_task not in {"", task}:
            failure = f"measurement task 与 fixture 不一致: measurement={measurement_task} fixture={task}"
        if failure is not None:
            provenance_failures.append(f"request_id={request_id}: {failure}")
            continue

        eval_measurements.append(measurement)
        quality_records.append(_quality_record_evidence(measurement, fixture_row))

    profile_means = {
        profile: sum(losses) / len(losses)
        for profile, losses in _losses_by_profile(eval_measurements).items()
        if losses
    }
    grouped_losses: dict[str, dict[str, list[float]]] = {}
    for record in quality_records:
        risk_family = str(record["risk_family"])
        profile = str(record["profile"])
        grouped_losses.setdefault(risk_family, {}).setdefault(profile, []).append(float(record["quality_loss"]))
    group_means = {
        risk_family: {
            profile: sum(losses) / len(losses)
            for profile, losses in sorted(profile_losses.items())
            if losses
        }
        for risk_family, profile_losses in sorted(grouped_losses.items())
    }

    sensitive_profiles = {
        "kivi_sensitive": MAINSTREAM_KIVI_PROFILES,
        "h2o_sensitive": MAINSTREAM_H2O_PROFILES,
    }
    task_coverage = {risk_family: set() for risk_family in sensitive_profiles}
    for record in quality_records:
        risk_family = str(record["risk_family"])
        if risk_family in sensitive_profiles and str(record["profile"]) in sensitive_profiles[risk_family]:
            task_coverage[risk_family].add(str(record["task"]))

    qualifying_profiles: list[str] = []
    sensitive_control_gaps: dict[str, float | None] = {}
    for risk_family, family_profiles in sensitive_profiles.items():
        sensitive_means = [
            group_means[risk_family][profile]
            for profile in family_profiles
            if profile in group_means.get(risk_family, {})
        ]
        control_means = [
            group_means["low_risk"][profile]
            for profile in family_profiles
            if profile in group_means.get("low_risk", {})
        ]
        sensitive_control_gaps[risk_family] = (
            max(sensitive_means) - max(control_means)
            if sensitive_means and control_means
            else None
        )
        qualifying_profiles.extend(
            profile
            for profile in family_profiles
            if group_means.get(risk_family, {}).get(profile, float("-inf")) > QUALITY_GATE_THRESHOLD
        )

    covered_tasks = set().union(*task_coverage.values())

    errors: list[str] = []
    if provenance_failures:
        errors.append(f"存在 {len(provenance_failures)} 条无法追溯到注入内容与模板的 eval 质量记录。")
    for risk_family, label in (("kivi_sensitive", "KIVI"), ("h2o_sensitive", "H2O")):
        if not any(profile in qualifying_profiles for profile in sensitive_profiles[risk_family]):
            errors.append(
                f"{label} 敏感组至少需要一个主流 profile 的 group mean quality loss > {QUALITY_GATE_THRESHOLD:.2f}。"
            )
        if task_coverage[risk_family] != {"qa", "summary"}:
            errors.append(f"{risk_family} 风险质量记录必须同时覆盖 QA 与 Summary。")
        gap = sensitive_control_gaps[risk_family]
        if gap is None:
            errors.append(f"{risk_family} 或 low_risk 对照组缺少同家族质量证据，无法计算风险差。")
        elif gap <= 0.0:
            errors.append(f"{risk_family} 相对 low_risk 对照组必须有正向风险差，当前为 {gap:.6f}。")

    return TraceQualityValidationResult(
        passed=not errors,
        errors=errors,
        qualifying_profiles=sorted(set(qualifying_profiles)),
        covered_tasks=covered_tasks,
        profile_means={key: round(value, 6) for key, value in sorted(profile_means.items())},
        group_means={
            risk_family: {profile: round(value, 6) for profile, value in sorted(profile_values.items())}
            for risk_family, profile_values in sorted(group_means.items())
        },
        task_coverage=task_coverage,
        sensitive_control_gaps={
            risk_family: None if value is None else round(value, 6)
            for risk_family, value in sorted(sensitive_control_gaps.items())
        },
        provenance_failures=provenance_failures,
        quality_records=quality_records,
    )


def _losses_by_profile(measurements: list[ProfileMeasurement]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in measurements:
        if row.quality_loss is None:
            continue
        grouped.setdefault(row.profile, []).append(float(row.quality_loss))
    return grouped


def _fixture_request_id(row: ProfileMeasurement) -> str:
    original = row.extra.get("original_request_id")
    if original not in {None, ""}:
        return str(original)
    request_id = str(row.request_id)
    return request_id.split("__pressure", 1)[0]


def _quality_record_provenance_failure(fixture_row: dict[str, Any]) -> str | None:
    metadata = fixture_row.get("metadata")
    if not isinstance(metadata, dict):
        return "metadata 必须是对象"
    required_values = {
        "source": "hybrid_session_builder",
        "source_dataset": "sharegpt_longbench_hybrid_session",
        "content_source_dataset": "longbench",
    }
    for key, expected in required_values.items():
        if str(metadata.get(key, "")).strip().lower() != expected:
            return f"{key} 必须是 {expected}"
    for key in (
        "content_source_request_id",
        "content_source_index",
        "injection_template",
        "original_session_id",
        "hybrid_turn_role",
    ):
        value = metadata.get(key)
        if value is None or value == "":
            return f"metadata 缺少 {key}"
    try:
        turn_index = int(fixture_row.get("turn_index", -1))
    except (TypeError, ValueError):
        return "turn_index 必须是整数"
    if turn_index < 2:
        return f"质量记录不能来自 chat 开场 turn: turn_index={turn_index}"
    if str(fixture_row.get("task", "")).strip().lower() not in {"qa", "summary"}:
        return "质量记录 task 必须是 QA/Summary"
    return None


def _quality_record_evidence(
    measurement: ProfileMeasurement,
    fixture_row: dict[str, Any],
) -> dict[str, Any]:
    metadata = fixture_row["metadata"]
    return {
        "measurement_request_id": str(measurement.request_id),
        "fixture_request_id": str(fixture_row["request_id"]),
        "session_id": str(fixture_row.get("session_id", "")),
        "turn_index": int(fixture_row.get("turn_index", 0)),
        "task": str(fixture_row["task"]).strip().lower(),
        "risk_family": str(metadata["risk_family"]),
        "profile": str(measurement.profile),
        "quality_loss": float(measurement.quality_loss),
        "content_source_dataset": str(metadata["content_source_dataset"]),
        "content_source_request_id": str(metadata["content_source_request_id"]),
        "content_source_index": metadata["content_source_index"],
        "injection_template": str(metadata["injection_template"]),
        "original_session_id": str(metadata["original_session_id"]),
        "hybrid_turn_role": str(metadata["hybrid_turn_role"]),
    }


def _load_fixture_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
