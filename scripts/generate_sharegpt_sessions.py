from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_source_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty source file: {path}")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict)]


def _normalize_conversation(row: dict) -> list[dict[str, str]]:
    messages = row.get("conversation") or row.get("conversations")
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if "human" in item and "assistant" in item:
            human = str(item.get("human") or "").strip()
            assistant = str(item.get("assistant") or "").strip()
            if human:
                normalized.append({"from": "human", "value": human})
            if assistant:
                normalized.append({"from": "gpt", "value": assistant})
            continue
        speaker = str(item.get("from") or item.get("role") or "").strip().lower()
        value = str(item.get("value") or item.get("content") or "").strip()
        if not value:
            continue
        if speaker in {"human", "user"}:
            normalized.append({"from": "human", "value": value})
        elif speaker in {"assistant", "gpt"}:
            normalized.append({"from": "gpt", "value": value})
    return normalized


def build_sessions(rows: list[dict], *, min_user_turns: int) -> list[dict]:
    sessions: list[dict] = []
    user_turns = 0
    for row in rows:
        conversation = _normalize_conversation(row)
        if not conversation:
            continue
        user_turn_count = sum(1 for item in conversation if item["from"] == "human")
        if user_turn_count == 0:
            continue
        sessions.append(
            {
                "id": f"session_{len(sessions):03d}",
                "conversations": conversation,
            }
        )
        user_turns += user_turn_count
        if user_turns >= min_user_turns:
            break
    if user_turns < min_user_turns:
        raise ValueError(f"could only collect {user_turns} user turns, below target {min_user_turns}")
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ShareGPT session fixture with 200+ user turns.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-user-turns", type=int, default=240)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    rows = _iter_source_rows(source)
    sessions = build_sessions(rows, min_user_turns=args.min_user_turns)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    user_turns = sum(
        1
        for session in sessions
        for item in session["conversations"]
        if item["from"] == "human"
    )
    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(output),
                "sessions": len(sessions),
                "user_turns": user_turns,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
