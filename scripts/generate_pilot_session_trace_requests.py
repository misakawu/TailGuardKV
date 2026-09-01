#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from run_util.core_types import ProfileMeasurement
from run_util.io_utils import read_measurements


OUTPUT_PATH = REPO_ROOT / "data" / "fixtures" / "pilot_session_trace_requests.jsonl"
VALIDATION_OUTPUT_PATH = REPO_ROOT / "data" / "golden" / "split_validation_v1.html"
TOTAL_REQUESTS = 240
SESSION_TURNS = 5
MEMORY_TURNS = 3
QUALITY_TURNS = 2
DEFAULT_CALIBRATION_RATIO = 0.6
KS_FAILURE_THRESHOLD = 0.1
MAINSTREAM_KIVI_PROFILES = (
    "kivi_2bit_residual16",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
)
MAINSTREAM_H2O_PROFILES = (
    "h2o_heavy05_recent05",
    "h2o_heavy08_recent08",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
LOW_RISK_PROFILES = (
    "kivi_4bit_residual64",
    "h2o_heavy10_recent10",
)


@dataclass(frozen=True)
class SplitValidationMetric:
    group: str
    profile: str
    calibration_count: int
    eval_count: int
    ks: float
    p95_delta: float
    calibration_cdf: list[dict[str, float]] = field(default_factory=list)
    eval_cdf: list[dict[str, float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "profile": self.profile,
            "calibration_count": self.calibration_count,
            "eval_count": self.eval_count,
            "ks": self.ks,
            "p95_delta": self.p95_delta,
            "calibration_cdf": self.calibration_cdf,
            "eval_cdf": self.eval_cdf,
        }


@dataclass(frozen=True)
class SplitValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    metrics: list[SplitValidationMetric] = field(default_factory=list)
    html: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "metrics": [metric.to_json() for metric in self.metrics],
            "html": self.html,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pilot session trace fixture with risk-stratified split.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--measurements", default="")
    parser.add_argument("--validation-output", default=str(VALIDATION_OUTPUT_PATH))
    args = parser.parse_args()

    templates = generate_templates()
    request_risk_families = _request_risk_family_lookup(templates)
    risk_lookup = None
    if args.measurements:
        risk_lookup = build_split_risk_lookup(
            read_measurements(Path(args.measurements)),
            request_risk_families=request_risk_families,
        )
    rows = build_requests_from_templates(templates, risk_lookup=risk_lookup)
    validation = validate_split_balance(rows, risk_lookup)
    if args.validation_output:
        validation_path = Path(args.validation_output)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(validation.html, encoding="utf-8")
    if not validation.passed:
        print(json.dumps({"errors": validation.errors}, ensure_ascii=False, sort_keys=True))
        return 2
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output_path), "rows": len(rows), "validation_output": str(args.validation_output)},
            ensure_ascii=False,
        )
    )
    return 0


def generate_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    qa_high_risk_sessions = 12
    qa_low_risk_sessions = 12
    summary_high_risk_sessions = 12
    summary_low_risk_sessions = 12

    for index in range(qa_high_risk_sessions):
        templates.extend(_qa_session(index, risk_family="kivi_sensitive"))
    for index in range(qa_low_risk_sessions):
        templates.extend(_qa_session(index + qa_high_risk_sessions, risk_family="low_risk"))
    for index in range(summary_high_risk_sessions):
        templates.extend(_summary_session(index, risk_family="h2o_sensitive"))
    for index in range(summary_low_risk_sessions):
        templates.extend(_summary_session(index + summary_high_risk_sessions, risk_family="low_risk"))

    if len(templates) != TOTAL_REQUESTS:
        raise RuntimeError(f"session trace 模板数量错误: {len(templates)} != {TOTAL_REQUESTS}")
    return templates


def build_requests_from_templates(
    templates: list[dict[str, Any]],
    risk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    normalized = [_normalize_template(row) for row in templates]
    ordered = sorted(
        normalized,
        key=lambda row: (
            0 if not row["metadata"].get("followup_kind") else 1,
            0 if int(row.get("turn_index", 0)) < MEMORY_TURNS else 1,
            str(row.get("task", "")),
            str(row.get("session_id", "")),
            int(row.get("turn_index", 0)),
            str(row.get("request_id", "")),
        ),
    )
    stratified = assign_stratified_splits(ordered, risk_lookup)
    memory_cutoff = int(len(ordered) * 0.6)
    for arrival_index, row in enumerate(stratified):
        row["metadata"]["pressure_phase"] = "memory" if arrival_index < memory_cutoff else "quality"
        if row["metadata"]["pressure_phase"] == "memory":
            row["metadata"]["followup_kind"] = ""
        row["metadata"]["arrival_index"] = arrival_index
    return stratified


def build_split_risk_lookup(
    measurements: list[ProfileMeasurement],
    *,
    request_risk_families: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for measurement in measurements:
        request_id = _measurement_request_id(measurement)
        risk_family = _measurement_risk_family(measurement, request_risk_families or {})
        entry = lookup.setdefault(
            request_id,
            {
                "split_score": 0.0,
                "split_score_family": risk_family,
                "profile_losses": {},
            },
        )
        if measurement.quality_loss is None:
            continue
        loss = float(measurement.quality_loss)
        profile_losses = entry["profile_losses"]
        profile_losses[measurement.profile] = max(float(profile_losses.get(measurement.profile, 0.0)), loss)
        entry["split_score"] = _recompute_split_score(
            str(entry.get("split_score_family", risk_family)),
            profile_losses,
        )
    return lookup


def assign_stratified_splits(
    rows: list[dict[str, Any]],
    risk_lookup: dict[str, dict[str, Any]] | None = None,
    *,
    calibration_ratio: float = DEFAULT_CALIBRATION_RATIO,
) -> list[dict[str, Any]]:
    lookup = risk_lookup or _metadata_risk_lookup(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    for group_rows in grouped.values():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                -_request_split_score(row, lookup),
                str(row["metadata"].get("risk_family", "")),
                str(row["request_id"]),
            ),
        )
        target_calibration = max(1, int(round(len(ranked) * calibration_ratio)))
        calibration_assigned = 0
        for index, row in enumerate(ranked):
            # Spread calibration quota across the ranked list so the 60/40 ratio
            # does not get "patched" by moving a low-risk tail block wholesale.
            ideal_calibration = ((index + 1) * target_calibration) / len(ranked)
            split = "calibration" if calibration_assigned < ideal_calibration else "eval"
            if split == "calibration":
                calibration_assigned += 1
            row["metadata"]["split"] = split
            row["metadata"]["split_score"] = round(_request_split_score(row, lookup), 6)
    return rows


def validate_split_balance(
    rows: list[dict[str, Any]],
    risk_lookup: dict[str, dict[str, Any]] | None = None,
    *,
    ks_threshold: float = KS_FAILURE_THRESHOLD,
) -> SplitValidationResult:
    lookup = risk_lookup or _metadata_risk_lookup(rows)
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        split = str(row["metadata"].get("split", "eval"))
        group_key = _group_key(row)
        profile_losses = _profile_losses_for_row(row, lookup)
        for profile, loss in profile_losses.items():
            slot = grouped.setdefault(group_key, {}).setdefault(profile, {"calibration": [], "eval": []})
            slot["calibration" if split == "calibration" else "eval"].append(float(loss))

    errors: list[str] = []
    metrics: list[SplitValidationMetric] = []
    for group_key in sorted(grouped):
        group_label = f"{group_key[0]}/{group_key[1]}"
        for profile in sorted(grouped[group_key]):
            calibration_values = sorted(grouped[group_key][profile]["calibration"])
            eval_values = sorted(grouped[group_key][profile]["eval"])
            if not calibration_values or not eval_values:
                errors.append(f"{group_label} {profile} 缺少 calibration/eval 双侧样本。")
                continue
            ks = _ks_statistic(calibration_values, eval_values)
            p95_delta = abs(_percentile(calibration_values, 0.95) - _percentile(eval_values, 0.95))
            metric = SplitValidationMetric(
                group=group_label,
                profile=profile,
                calibration_count=len(calibration_values),
                eval_count=len(eval_values),
                ks=round(ks, 6),
                p95_delta=round(p95_delta, 6),
                calibration_cdf=_cdf_points(calibration_values),
                eval_cdf=_cdf_points(eval_values),
            )
            metrics.append(metric)
            if ks > ks_threshold:
                errors.append(f"{group_label} {profile} 的 KS={ks:.4f} 超过阈值 {ks_threshold:.2f}。")
    html = _render_split_validation_html(metrics, errors, ks_threshold)
    return SplitValidationResult(passed=not errors, errors=errors, metrics=metrics, html=html)


def _normalize_template(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    history_turns = list(row.get("history_turns") or [])
    metadata.setdefault("split", "eval")
    metadata.setdefault("length_bucket", "medium")
    metadata.setdefault("risk_family", "low_risk")
    metadata.setdefault("risk_profiles", _risk_profiles(metadata["risk_family"]))
    metadata.setdefault("pressure_phase", "memory")
    metadata.setdefault("followup_kind", "")
    return {
        "request_id": str(row["request_id"]),
        "task": str(row["task"]),
        "prompt": str(row["prompt"]),
        "reference": str(row["reference"]),
        "session_id": str(row["session_id"]),
        "turn_index": int(row.get("turn_index", 0)),
        "history_turns": history_turns,
        "metadata": metadata,
    }


def _metadata_risk_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        risk_family = str(metadata.get("risk_family", "low_risk"))
        followup_kind = str(metadata.get("followup_kind", ""))
        pressure_phase = str(metadata.get("pressure_phase", "memory"))
        if risk_family == "low_risk":
            base_score = 0.02
        elif followup_kind:
            base_score = 0.35
        elif pressure_phase == "quality":
            base_score = 0.25
        else:
            base_score = 0.18
        profile_losses = {str(profile): base_score for profile in metadata.get("risk_profiles") or []}
        lookup[str(row["request_id"])] = {
            "split_score": _recompute_split_score(risk_family, profile_losses),
            "split_score_family": risk_family,
            "profile_losses": profile_losses,
        }
    return lookup


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    return (str(row.get("task", "")), str(metadata.get("length_bucket", "unknown")))


def _request_split_score(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> float:
    request_id = str(row["request_id"])
    entry = lookup.get(request_id)
    if not entry:
        return 0.0
    return float(entry.get("split_score", 0.0))


def _profile_losses_for_row(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, float]:
    request_id = str(row["request_id"])
    entry = lookup.get(request_id) or {}
    profile_losses = entry.get("profile_losses")
    if isinstance(profile_losses, dict) and profile_losses:
        return {str(profile): float(loss) for profile, loss in profile_losses.items()}
    score = float(entry.get("split_score", 0.0))
    return {profile: score for profile in row["metadata"].get("risk_profiles") or []}


def _measurement_request_id(measurement: ProfileMeasurement) -> str:
    original = measurement.extra.get("original_request_id")
    if original not in {None, ""}:
        return str(original)
    request_id = str(measurement.request_id)
    return request_id.split("__pressure", 1)[0]


def _is_lossy_profile(profile: str) -> bool:
    return not profile.startswith("full")


def _request_risk_family_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["request_id"]): str((row.get("metadata") or {}).get("risk_family", "low_risk"))
        for row in rows
    }


def _measurement_risk_family(measurement: ProfileMeasurement, request_risk_families: dict[str, str]) -> str:
    request_id = _measurement_request_id(measurement)
    if request_id in request_risk_families:
        return str(request_risk_families[request_id])
    task = str(measurement.extra.get("task", "")).lower()
    if task == "qa":
        return "kivi_sensitive"
    if task == "summary":
        return "h2o_sensitive"
    return "low_risk"


def _recompute_split_score(risk_family: str, profile_losses: dict[str, Any]) -> float:
    relevant_losses = [
        float(loss)
        for profile, loss in profile_losses.items()
        if _profile_relevant_to_risk_family(str(profile), risk_family)
    ]
    if not relevant_losses:
        return 0.0
    return max(relevant_losses)


def _profile_relevant_to_risk_family(profile: str, risk_family: str) -> bool:
    if not _is_lossy_profile(profile):
        return False
    if risk_family == "kivi_sensitive":
        return profile.startswith("kivi_")
    if risk_family == "h2o_sensitive":
        return profile.startswith("h2o_")
    return True


def _ks_statistic(lhs: list[float], rhs: list[float]) -> float:
    points = sorted(set(lhs + rhs))
    max_gap = 0.0
    for point in points:
        lhs_cdf = sum(1 for value in lhs if value <= point) / len(lhs)
        rhs_cdf = sum(1 for value in rhs if value <= point) / len(rhs)
        max_gap = max(max_gap, abs(lhs_cdf - rhs_cdf))
    return max_gap


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _cdf_points(values: list[float]) -> list[dict[str, float]]:
    return [
        {"x": round(value, 6), "y": round((index + 1) / len(values), 6)}
        for index, value in enumerate(sorted(values))
    ]


def _render_split_validation_html(
    metrics: list[SplitValidationMetric],
    errors: list[str],
    ks_threshold: float,
) -> str:
    rows = "\n".join(
        (
            "<tr>"
            f"<td>{metric.group}</td>"
            f"<td>{metric.profile}</td>"
            f"<td>{metric.calibration_count}</td>"
            f"<td>{metric.eval_count}</td>"
            f"<td>{metric.ks:.6f}</td>"
            f"<td>{metric.p95_delta:.6f}</td>"
            f"<td><pre>{json.dumps(metric.calibration_cdf, ensure_ascii=False)}</pre></td>"
            f"<td><pre>{json.dumps(metric.eval_cdf, ensure_ascii=False)}</pre></td>"
            "</tr>"
        )
        for metric in metrics
    )
    error_items = "".join(f"<li>{error}</li>" for error in errors) or "<li>none</li>"
    return (
        "<html><head><meta charset='utf-8'><title>split validation</title></head><body>"
        "<h1>Split Validation v1</h1>"
        f"<p>KS threshold: {ks_threshold:.2f}</p>"
        "<h2>Errors</h2>"
        f"<ul>{error_items}</ul>"
        "<h2>Metrics</h2>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>group</th><th>profile</th><th>cal_count</th><th>eval_count</th><th>KS</th><th>p95_delta</th><th>calibration_cdf</th><th>eval_cdf</th></tr>"
        f"{rows}"
        "</table>"
        "</body></html>"
    )


def _qa_session(session_index: int, *, risk_family: str) -> list[dict[str, Any]]:
    session_id = f"qa-session-{session_index:03d}"
    entity = ["Atlas", "Borealis", "Cinder", "Delta"][session_index % 4]
    base_day = 12 + (session_index % 7)
    quantity = 18 + session_index
    location = ["warehouse-7", "checkpoint-3", "relay-5", "depot-2"][session_index % 4]
    return [
        _request_row(
            request_id=f"{session_id}-turn0",
            task="qa",
            session_id=session_id,
            turn_index=0,
            prompt=(
                f"Context: Team {entity} can ship exactly {quantity} crates on August {base_day}. "
                f"The destination must remain {location}. Keep those constraints for later.\n"
                "Question: Repeat the shipping plan in one sentence.\nAnswer:"
            ),
            reference=f"Team {entity} ships {quantity} crates on August {base_day} to {location}.",
            history_turns=[],
            split="calibration" if risk_family != "low_risk" else "calibration",
            length_bucket="medium",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn1",
            task="qa",
            session_id=session_id,
            turn_index=1,
            prompt="Question: What date was locked for the shipment, and what cannot change?\nAnswer:",
            reference=f"August {base_day}; the destination {location} cannot change.",
            history_turns=[
                f"User: shipping plan for team {entity}",
                f"Assistant: Team {entity} ships {quantity} crates on August {base_day} to {location}.",
            ],
            split="calibration",
            length_bucket="medium",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn2",
            task="qa",
            session_id=session_id,
            turn_index=2,
            prompt=(
                "Question: A coordinator asks whether the shipment can be split into two equal batches "
                "without changing the original date or destination. Answer yes/no and keep the constraints.\nAnswer:"
            ),
            reference=(
                f"Yes. Split {quantity} crates into two equal batches while keeping August {base_day} "
                f"and destination {location} unchanged."
            ),
            history_turns=[
                f"User: shipping plan for team {entity}",
                f"Assistant: Team {entity} ships {quantity} crates on August {base_day} to {location}.",
                "User: restate locked date and destination",
                f"Assistant: August {base_day}; destination {location} stays fixed.",
            ],
            split="eval",
            length_bucket="medium",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn3",
            task="qa",
            session_id=session_id,
            turn_index=3,
            prompt=(
                "Follow-up: After cache pressure, recover the earlier constraints and answer: "
                "how many crates are in each batch, on which date, and to which destination?\nAnswer:"
            ),
            reference=f"{quantity // 2} crates per batch on August {base_day} to {location}.",
            history_turns=[
                f"User: shipping plan for team {entity}",
                f"Assistant: Team {entity} ships {quantity} crates on August {base_day} to {location}.",
                "User: can the shipment be split?",
                f"Assistant: Yes. Keep August {base_day} and {location}; each batch has {quantity // 2} crates.",
            ],
            split="eval",
            length_bucket="medium",
            risk_family=risk_family,
            pressure_phase="quality",
            followup_kind="constraint_recall",
        ),
        _request_row(
            request_id=f"{session_id}-turn4",
            task="qa",
            session_id=session_id,
            turn_index=4,
            prompt=(
                "Final check: State the full shipment plan exactly, including the team name, crate count, date, "
                "destination, and the two-batch constraint.\nAnswer:"
            ),
            reference=(
                f"Team {entity} ships {quantity} crates to {location} on August {base_day}, "
                f"split into two batches of {quantity // 2}."
            ),
            history_turns=[
                f"User: shipping plan for team {entity}",
                f"Assistant: Team {entity} ships {quantity} crates on August {base_day} to {location}.",
                "User: recover batch, date, destination",
                f"Assistant: {quantity // 2} crates per batch on August {base_day} to {location}.",
            ],
            split="eval",
            length_bucket="medium",
            risk_family=risk_family,
            pressure_phase="quality",
            followup_kind="constraint_recall",
        ),
    ]


def _summary_session(session_index: int, *, risk_family: str) -> list[dict[str, Any]]:
    session_id = f"summary-session-{session_index:03d}"
    district = ["North Basin", "River Ward", "Port Annex", "Hill Sector"][session_index % 4]
    metric_a = 40 + session_index
    metric_b = 17 + (session_index % 9)
    metric_c = 3 + (session_index % 5)
    document = (
        f"Report for {district}. Quarter one backlog fell by {metric_a}% after a staffing reshuffle. "
        f"Quarter two still missed the response SLA by {metric_b} hours because one routing table stayed stale. "
        f"Quarter three added a three-step escalation: intake audit, supervisor review, and nightly replay. "
        f"The report warns that only {metric_c} satellite offices completed the replay drill, "
        "so the main recommendation is to preserve the escalation details for the next audit."
    )
    return [
        _request_row(
            request_id=f"{session_id}-turn0",
            task="summary",
            session_id=session_id,
            turn_index=0,
            prompt=f"Summarize the report with the main metric changes and the final recommendation.\n\n{document}\n\nSummary:",
            reference=(
                f"{district} reduced backlog by {metric_a}% but still missed the SLA by {metric_b} hours; "
                f"the report recommends preserving the intake audit, supervisor review, and nightly replay details."
            ),
            history_turns=[],
            split="calibration" if risk_family != "low_risk" else "calibration",
            length_bucket="long",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn1",
            task="summary",
            session_id=session_id,
            turn_index=1,
            prompt="Summarize only the operational sequence introduced in quarter three.\nSummary:",
            reference="The quarter-three sequence is intake audit, supervisor review, then nightly replay.",
            history_turns=[
                f"User: summarize report for {district}",
                f"Assistant: backlog down {metric_a}%, SLA miss {metric_b} hours, preserve escalation details.",
            ],
            split="calibration",
            length_bucket="long",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn2",
            task="summary",
            session_id=session_id,
            turn_index=2,
            prompt="Summarize the remaining risk after the process change, naming the weak detail explicitly.\nSummary:",
            reference=f"The weak detail is that only {metric_c} satellite offices completed the replay drill.",
            history_turns=[
                f"User: summarize report for {district}",
                f"Assistant: backlog down {metric_a}%, SLA miss {metric_b} hours, preserve escalation details.",
                "User: list the quarter-three sequence",
                "Assistant: intake audit, supervisor review, nightly replay.",
            ],
            split="eval",
            length_bucket="long",
            risk_family=risk_family,
            pressure_phase="memory",
        ),
        _request_row(
            request_id=f"{session_id}-turn3",
            task="summary",
            session_id=session_id,
            turn_index=3,
            prompt=(
                "Follow-up after pressure: write a two-sentence summary that keeps the backlog change, the SLA miss, "
                "and the exact three-step escalation sequence.\nSummary:"
            ),
            reference=(
                f"{district} cut backlog by {metric_a}% but still missed the SLA by {metric_b} hours. "
                "The escalation sequence remains intake audit, supervisor review, and nightly replay."
            ),
            history_turns=[
                f"User: summarize report for {district}",
                f"Assistant: backlog down {metric_a}%, SLA miss {metric_b} hours, preserve escalation details.",
                "User: summarize the quarter-three sequence",
                "Assistant: intake audit, supervisor review, nightly replay.",
            ],
            split="eval",
            length_bucket="long",
            risk_family=risk_family,
            pressure_phase="quality",
            followup_kind="detail_recall",
        ),
        _request_row(
            request_id=f"{session_id}-turn4",
            task="summary",
            session_id=session_id,
            turn_index=4,
            prompt=(
                "Final follow-up: summarize the report and explicitly mention how many satellite offices completed "
                "the replay drill.\nSummary:"
            ),
            reference=(
                f"{district} cut backlog by {metric_a}%, still missed the SLA by {metric_b} hours, "
                f"and only {metric_c} satellite offices completed the replay drill."
            ),
            history_turns=[
                f"User: summarize report for {district}",
                f"Assistant: backlog down {metric_a}%, SLA miss {metric_b} hours, preserve escalation details.",
                "User: keep the exact escalation details",
                "Assistant: intake audit, supervisor review, nightly replay.",
            ],
            split="eval",
            length_bucket="long",
            risk_family=risk_family,
            pressure_phase="quality",
            followup_kind="detail_recall",
        ),
    ]


def _request_row(
    *,
    request_id: str,
    task: str,
    session_id: str,
    turn_index: int,
    prompt: str,
    reference: str,
    history_turns: list[str],
    split: str,
    length_bucket: str,
    risk_family: str,
    pressure_phase: str,
    followup_kind: str = "",
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "task": task,
        "prompt": prompt,
        "reference": reference,
        "session_id": session_id,
        "turn_index": turn_index,
        "history_turns": history_turns,
        "metadata": {
            "split": split,
            "length_bucket": length_bucket,
            "risk_family": risk_family,
            "risk_profiles": list(_risk_profiles(risk_family)),
            "pressure_phase": pressure_phase,
            "followup_kind": followup_kind,
            "source_dataset": "pilot_session_trace_generator",
        },
    }


def _risk_profiles(risk_family: str) -> tuple[str, ...]:
    if risk_family == "kivi_sensitive":
        return MAINSTREAM_KIVI_PROFILES
    if risk_family == "h2o_sensitive":
        return MAINSTREAM_H2O_PROFILES
    return LOW_RISK_PROFILES


if __name__ == "__main__":
    raise SystemExit(main())
