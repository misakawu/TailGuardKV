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

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["covered_tasks"] = sorted(self.covered_tasks)
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
    fixture_by_id = {str(row["request_id"]): row for row in fixture_rows}
    eval_request_ids = {
        request_id
        for request_id, row in fixture_by_id.items()
        if str((row.get("metadata") or {}).get("split", "")) == "eval"
    }
    eval_measurements = [
        row
        for row in measurements
        if _fixture_request_id(row) in eval_request_ids and str(row.extra.get("split", "")) == "eval" and row.quality_loss is not None
    ]
    profile_means = {
        profile: sum(losses) / len(losses)
        for profile, losses in _losses_by_profile(eval_measurements).items()
        if losses
    }
    qualifying_profiles = [profile for profile, value in profile_means.items() if value > QUALITY_GATE_THRESHOLD]
    qualifying_h2o = [profile for profile in qualifying_profiles if profile in MAINSTREAM_H2O_PROFILES]
    qualifying_kivi = [profile for profile in qualifying_profiles if profile in MAINSTREAM_KIVI_PROFILES]
    covered_tasks = {
        str(fixture_by_id[_fixture_request_id(row)]["task"])
        for row in eval_measurements
        if float(row.quality_loss or 0.0) > 0.0 and _fixture_request_id(row) in fixture_by_id
    }

    errors: list[str] = []
    if len(qualifying_profiles) < 2:
        errors.append(
            f"至少需要 2 个不同 profile 的 mean_quality_loss > {QUALITY_GATE_THRESHOLD:.2f}，当前只有 {len(qualifying_profiles)} 个。"
        )
    if not qualifying_h2o:
        errors.append("缺少满足阈值的 H2O 主流档位 profile。")
    if not qualifying_kivi:
        errors.append("缺少满足阈值的 KIVI 主流档位 profile。")
    if covered_tasks != {"qa", "summary"}:
        errors.append("非零 loss 来源请求必须同时覆盖 QA 与 Summary。")

    return TraceQualityValidationResult(
        passed=not errors,
        errors=errors,
        qualifying_profiles=sorted(qualifying_profiles),
        covered_tasks=covered_tasks,
        profile_means={key: round(value, 6) for key, value in sorted(profile_means.items())},
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


def _load_fixture_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
