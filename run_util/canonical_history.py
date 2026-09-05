"""Canonical assistant-history contracts for measured multi-turn sessions."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


CANONICAL_HISTORY_MODE = "full_gpu_generated_v1"
CANONICAL_HISTORY_SOURCE_PROFILE = "full_gpu"
CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS = 16


class CanonicalHistoryError(ValueError):
    """Raised when a canonical fixture cannot safely be replayed."""


@dataclass(frozen=True)
class CanonicalBootstrapManifest:
    """Immutable provenance for a full-GPU generated canonical fixture."""

    schema_version: int
    fixture_hash: str
    bootstrap_table_hash: str
    profile: str
    generation_max_new_tokens: int
    turn_history_hashes: dict[str, str]
    output_token_counts: dict[str, list[int]]

    @classmethod
    def from_fixture(cls, rows: Iterable[dict[str, Any]], *, bootstrap_table_hash: str) -> "CanonicalBootstrapManifest":
        copied = [deepcopy(row) for row in rows]
        validate_canonical_history_rows(copied)
        if not copied:
            raise CanonicalHistoryError("canonical_history_mismatch: empty canonical fixture")
        hashes = {str(_metadata(row).get("canonical_bootstrap_table_hash") or "") for row in copied}
        if hashes != {str(bootstrap_table_hash)}:
            raise CanonicalHistoryError("canonical_history_mismatch: manifest bootstrap table hash mismatch")
        return cls(
            schema_version=1,
            fixture_hash=_fixture_hash(copied),
            bootstrap_table_hash=str(bootstrap_table_hash),
            profile=CANONICAL_HISTORY_SOURCE_PROFILE,
            generation_max_new_tokens=CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS,
            turn_history_hashes={str(row["request_id"]): str(_metadata(row)["canonical_history_hash"]) for row in copied},
            output_token_counts={str(row["request_id"]): list(_metadata(row)["canonical_history_output_token_counts"]) for row in copied},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_fixture(self, rows: Iterable[dict[str, Any]]) -> None:
        copied = [deepcopy(row) for row in rows]
        validate_canonical_history_rows(copied)
        if self.schema_version != 1 or self.profile != CANONICAL_HISTORY_SOURCE_PROFILE:
            raise CanonicalHistoryError("canonical_history_mismatch: unsupported canonical manifest")
        if self.generation_max_new_tokens != CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS:
            raise CanonicalHistoryError("canonical_history_mismatch: manifest generation bound must be 16")
        if self.fixture_hash != _fixture_hash(copied):
            raise CanonicalHistoryError("canonical_history_mismatch: manifest fixture hash mismatch")
        if any(str(_metadata(row).get("canonical_bootstrap_table_hash") or "") != self.bootstrap_table_hash for row in copied):
            raise CanonicalHistoryError("canonical_history_mismatch: manifest bootstrap table hash mismatch")
        actual = {str(row["request_id"]): str(_metadata(row)["canonical_history_hash"]) for row in copied}
        if actual != self.turn_history_hashes:
            raise CanonicalHistoryError("canonical_history_mismatch: manifest turn history hashes mismatch")


def canonical_history_hash(history: Iterable[str]) -> str:
    payload = json.dumps(list(history), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fixture_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_canonical_fixture(
    rows: Iterable[dict[str, Any]],
    bootstrap_outputs: dict[str, str],
    *,
    bootstrap_table_hash: str,
    output_token_count: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Attach generated full-GPU histories without changing scoring references."""
    copied = [deepcopy(row) for row in rows]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        session_id = str(row.get("session_id") or "")
        if not session_id:
            raise CanonicalHistoryError("canonical_history_mismatch: missing session_id")
        by_session[session_id].append(row)

    for session_id, session_rows in by_session.items():
        ordered = sorted(session_rows, key=_row_order)
        history: list[str] = []
        output_token_counts: list[int] = []
        for expected_turn, row in enumerate(ordered):
            turn_index = _turn_index(row)
            if turn_index != expected_turn:
                raise CanonicalHistoryError(
                    "canonical_history_mismatch: non-contiguous turn "
                    f"session={session_id} expected={expected_turn} actual={turn_index}"
                )
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "canonical_history_mode": CANONICAL_HISTORY_MODE,
                    "canonical_history": list(history),
                    "canonical_history_turns": turn_index,
                    "canonical_history_hash": canonical_history_hash(history),
                    "canonical_history_output_token_counts": list(output_token_counts),
                    "canonical_history_source_profile": CANONICAL_HISTORY_SOURCE_PROFILE,
                    "canonical_history_generation_max_new_tokens": CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS,
                    "canonical_bootstrap_table_hash": str(bootstrap_table_hash),
                }
            )
            row["metadata"] = metadata
            if expected_turn + 1 == len(ordered):
                continue
            request_id = str(row.get("request_id") or "")
            output = bootstrap_outputs.get(request_id)
            if output is None:
                raise CanonicalHistoryError(
                    f"canonical_history_mismatch: missing full_gpu output request_id={request_id}"
                )
            output = str(output)
            token_count = int(output_token_count(output))
            if token_count > CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS:
                raise CanonicalHistoryError(
                    "canonical_history_mismatch: full_gpu bootstrap output exceeds 16 tokens "
                    f"request_id={request_id} tokens={token_count}"
                )
            output_token_counts.append(token_count)
            history.extend((f"User: {row.get('prompt') or ''}", f"Assistant: {output}"))
    validate_canonical_history_rows(copied)
    return sorted(copied, key=lambda row: int(row.get("arrival_index") or 0))


def validate_canonical_history_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Reject missing, forged, cross-session, or reference-derived histories."""
    copied = [dict(row) for row in rows]
    canonical_rows = [row for row in copied if _metadata(row).get("canonical_history_mode")]
    if not canonical_rows:
        return
    if len(canonical_rows) != len(copied):
        raise CanonicalHistoryError("canonical_history_mismatch: mixed canonical and ordinary rows")

    bootstrap_hashes: set[str] = set()
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        metadata = _metadata(row)
        if metadata.get("canonical_history_mode") != CANONICAL_HISTORY_MODE:
            raise CanonicalHistoryError("canonical_history_mismatch: unsupported canonical history mode")
        if metadata.get("canonical_history_source_profile") != CANONICAL_HISTORY_SOURCE_PROFILE:
            raise CanonicalHistoryError("canonical_history_mismatch: canonical source profile must be full_gpu")
        if int(metadata.get("canonical_history_generation_max_new_tokens") or 0) != CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS:
            raise CanonicalHistoryError("canonical_history_mismatch: bootstrap generation bound must be 16")
        bootstrap_hash = str(metadata.get("canonical_bootstrap_table_hash") or "")
        if not bootstrap_hash:
            raise CanonicalHistoryError("canonical_history_mismatch: missing bootstrap table hash")
        bootstrap_hashes.add(bootstrap_hash)
        session_id = str(row.get("session_id") or "")
        if not session_id:
            raise CanonicalHistoryError("canonical_history_mismatch: missing session_id")
        by_session[session_id].append(row)
    if len(bootstrap_hashes) != 1:
        raise CanonicalHistoryError("canonical_history_mismatch: fixture mixes bootstrap tables")

    for session_id, session_rows in by_session.items():
        ordered = sorted(session_rows, key=_row_order)
        for expected_turn, row in enumerate(ordered):
            metadata = _metadata(row)
            if _turn_index(row) != expected_turn:
                raise CanonicalHistoryError(f"canonical_history_mismatch: invalid turn sequence session={session_id}")
            actual_history = metadata.get("canonical_history")
            if not isinstance(actual_history, list) or len(actual_history) != expected_turn * 2:
                raise CanonicalHistoryError(f"canonical_history_mismatch: history mismatch session={session_id} turn={expected_turn}")
            if int(metadata.get("canonical_history_turns", -1)) != expected_turn:
                raise CanonicalHistoryError(f"canonical_history_mismatch: history turn count session={session_id} turn={expected_turn}")
            output_token_counts = metadata.get("canonical_history_output_token_counts")
            if (
                not isinstance(output_token_counts, list)
                or len(output_token_counts) != expected_turn
                or any(
                    not isinstance(count, int)
                    or count < 0
                    or count > CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS
                    for count in output_token_counts
                )
            ):
                raise CanonicalHistoryError(
                    f"canonical_history_mismatch: bootstrap output token bound exceeds 16 session={session_id} turn={expected_turn}"
                )
            if str(metadata.get("canonical_history_hash") or "") != canonical_history_hash(actual_history):
                raise CanonicalHistoryError(f"canonical_history_mismatch: history hash session={session_id} turn={expected_turn}")
            for prior_turn, prior_row in enumerate(ordered[:expected_turn]):
                user = actual_history[prior_turn * 2]
                assistant = actual_history[prior_turn * 2 + 1]
                if user != f"User: {prior_row.get('prompt') or ''}" or not isinstance(assistant, str) or not assistant.startswith("Assistant: "):
                    raise CanonicalHistoryError(
                        f"canonical_history_mismatch: invalid history prefix session={session_id} turn={expected_turn}"
                    )


def canonical_history_for_row(row: dict[str, Any]) -> tuple[str, ...] | None:
    metadata = _metadata(row)
    if not metadata.get("canonical_history_mode"):
        return None
    history = metadata.get("canonical_history")
    if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
        raise CanonicalHistoryError("canonical_history_mismatch: canonical history is not a string list")
    return tuple(history)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _turn_index(row: dict[str, Any]) -> int:
    return int(row.get("turn_index") or 0)


def _row_order(row: dict[str, Any]) -> tuple[int, int, str]:
    return (_turn_index(row), int(row.get("arrival_index") or 0), str(row.get("request_id") or ""))
