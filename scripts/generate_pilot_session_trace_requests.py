#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_PATH = REPO_ROOT / "data" / "fixtures" / "pilot_session_trace_requests.jsonl"
TOTAL_REQUESTS = 240
SESSION_TURNS = 5
MEMORY_TURNS = 3
QUALITY_TURNS = 2
MAINSTREAM_KIVI_PROFILES = (
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
)
MAINSTREAM_H2O_PROFILES = (
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
LOW_RISK_PROFILES = (
    "kivi_4bit_residual64",
    "h2o_heavy10_recent10",
)


def main() -> int:
    rows = build_requests_from_templates(generate_templates())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "rows": len(rows)}, ensure_ascii=False))
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


def build_requests_from_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    memory_cutoff = int(len(ordered) * 0.6)
    for arrival_index, row in enumerate(ordered):
        row["metadata"]["pressure_phase"] = "memory" if arrival_index < memory_cutoff else "quality"
        if row["metadata"]["pressure_phase"] == "memory":
            row["metadata"]["followup_kind"] = ""
        row["metadata"]["arrival_index"] = arrival_index
    return ordered


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
