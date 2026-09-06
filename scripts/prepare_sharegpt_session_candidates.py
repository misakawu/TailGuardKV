#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    BASELINE_SESSION_MANIFEST,
    SHAREGPT_CANDIDATES,
    SHAREGPT_SOURCE,
    ensure_artifact_dirs,
    length_bucket,
    normalize_text,
    read_jsonl,
    write_json,
    write_jsonl,
)


CANONICAL_HISTORY_MODE = "full_gpu_generated_v1"
CANONICAL_ASSISTANT_PLACEHOLDER = "canonical " * 16


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ShareGPT session candidates for external labeling.")
    parser.add_argument("--input", default=str(SHAREGPT_SOURCE))
    parser.add_argument("--output", default=str(SHAREGPT_CANDIDATES))
    parser.add_argument("--manifest", default=str(BASELINE_SESSION_MANIFEST))
    parser.add_argument("--max-sessions", type=int, default=36)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=1024)
    parser.add_argument("--max-effective-prompt-chars", type=int, default=5000)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    candidates = build_session_candidates(
        rows,
        max_sessions=args.max_sessions,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_prompt_chars=args.max_prompt_chars,
        max_effective_prompt_chars=args.max_effective_prompt_chars,
    )
    output = Path(args.output)
    manifest = Path(args.manifest)
    ensure_artifact_dirs()
    write_jsonl(output, candidates)
    write_json(
        manifest,
        {
            "output": str(output),
            "rows": len(candidates),
            "sessions": len({row["session_id"] for row in candidates}),
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_prompt_chars": args.max_prompt_chars,
            "max_effective_prompt_chars": args.max_effective_prompt_chars,
            "canonical_history_mode": CANONICAL_HISTORY_MODE,
            "canonical_assistant_placeholder_tokens": 16,
            "source": str(args.input),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(candidates),
                "sessions": len({row["session_id"] for row in candidates}),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_session_candidates(
    rows: list[dict[str, Any]],
    *,
    max_sessions: int,
    min_turns: int,
    max_turns: int,
    max_prompt_chars: int,
    max_effective_prompt_chars: int = 4096,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("session_id"))].append(row)

    viable_sessions: list[tuple[int, int, str, list[dict[str, Any]]]] = []
    for session_id in sorted(grouped):
        if not session_id:
            continue
        ordered = sorted(
            grouped[session_id],
            key=lambda row: int(float(row.get("source_turn_index") or row.get("turn_id") or 0)),
        )
        if len(ordered) < min_turns:
            continue
        trimmed = ordered[:max_turns]
        worst_effective = _session_within_effective_prompt_budget(
            trimmed,
            max_prompt_chars=max_prompt_chars,
            max_effective_prompt_chars=max_effective_prompt_chars,
        )
        if worst_effective is None:
            continue
        # Prefer sessions that contain injected benchmark turns, then retain
        # longer effective context so risk-bearing examples are not discarded
        # by the old shortest-prompt-first heuristic.
        injected_turns = sum(1 for row in trimmed if _injected_dataset(row))
        viable_sessions.append((injected_turns, worst_effective, session_id, trimmed))
    selected_sessions = [
        rows
        for _, _, _, rows in sorted(
            viable_sessions,
            key=lambda item: (-item[0], -item[1], item[2]),
        )[:max_sessions]
    ]
    if len(selected_sessions) < max_sessions:
        raise RuntimeError(f"ShareGPT session 候选不足: {len(selected_sessions)} < {max_sessions}")

    exported: list[dict[str, Any]] = []
    arrival_index = 0
    for session_rows in selected_sessions:
        for turn_index, row in enumerate(session_rows):
            prompt = normalize_text(row.get("prompt"))[:max_prompt_chars]
            reference = normalize_text(row.get("reference"))
            if not prompt or not reference:
                continue
            dataset_config = normalize_text(row.get("dataset_config"))
            task = _task_for_row(row)
            metadata = {
                "source": "external_labeling_workspace",
                "source_dataset": "sharegpt_pilot",
                "split": "candidate",
                "risk_family": "unlabeled",
                "length_bucket": length_bucket(prompt),
                "dataset_config": dataset_config,
                "category": normalize_text(row.get("category")),
                "source_turn_index": normalize_text(row.get("source_turn_index")),
            }
            if _injected_dataset(row):
                metadata.update(
                    {
                        "injected_from": f"longbench_{dataset_config}",
                        "original_task": task,
                    }
                )
            exported.append(
                {
                    "request_id": normalize_text(row.get("request_id")) or f"{row['session_id']}_turn_{turn_index:03d}",
                    "task": task,
                    "prompt": prompt,
                    "reference": reference,
                    "session_id": normalize_text(row.get("session_id")),
                    "turn_index": turn_index,
                    "arrival_index": arrival_index,
                    "metadata": metadata,
                }
            )
            arrival_index += 1
    return exported


def _injected_dataset(row: dict[str, Any]) -> bool:
    config = normalize_text(row.get("dataset_config")).lower()
    return bool(config and config not in {"sharegpt", "common_en_70k"})


def _task_for_row(row: dict[str, Any]) -> str:
    config = normalize_text(row.get("dataset_config")).lower()
    if config in {"qasper", "gov_report", "qmsum", "multi_news", "summ_screen_fd"}:
        return "summary" if config != "qasper" else "qa"
    if config in {"lcc", "repobench-p", "repobench"}:
        return "code"
    return "chat"


def _session_within_effective_prompt_budget(
    rows: list[dict[str, Any]],
    *,
    max_prompt_chars: int,
    max_effective_prompt_chars: int,
) -> int | None:
    history: list[str] = []
    worst_effective = 0
    for row in rows:
        prompt = normalize_text(row.get("prompt"))[:max_prompt_chars]
        effective_prompt = "\n".join([*history, prompt]) if history else prompt
        worst_effective = max(worst_effective, len(effective_prompt))
        if len(effective_prompt) > max_effective_prompt_chars:
            return None
        history.append(f"User: {prompt}")
        # Candidate selection cannot use a reference as running history. The
        # actual full-GPU bootstrap output replaces this bounded placeholder.
        history.append(f"Assistant: {CANONICAL_ASSISTANT_PLACEHOLDER.strip()}")
    return worst_effective


if __name__ == "__main__":
    raise SystemExit(main())
