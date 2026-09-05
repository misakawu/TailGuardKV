#!/usr/bin/env python3
"""Bootstrap canonical history then replay only complete lossy profile tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from run_util.canonical_history import CanonicalBootstrapManifest, CanonicalHistoryError, validate_canonical_history_rows


PROFILES = (
    "full_gpu", "kivi_4bit_residual32", "kivi_4bit_residual64", "kivi_2bit_residual32",
    "kivi_2bit_residual64", "h2o_heavy10_recent10", "h2o_heavy15_recent15", "h2o_heavy20_recent20",
)
LOSSY_PROFILES = PROFILES[1:]


class CanonicalSupervisorError(ValueError):
    pass


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def load_input_rows(path: Path, *, expected_requests: int | None = None, max_requests: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected_requests is not None and len(rows) != expected_requests:
        raise CanonicalSupervisorError(f"canonical probe requires exactly {expected_requests} rows; got {len(rows)}")
    if max_requests is not None and max_requests < len(rows):
        raise CanonicalSupervisorError("canonical supervisor never truncates a probe input")
    return rows


def ensure_complete_replay(rows: Iterable[dict[str, Any]], *, request_ids: set[str], profiles: tuple[str, ...] = LOSSY_PROFILES) -> None:
    observed: dict[str, set[str]] = {request_id: set() for request_id in request_ids}
    for row in rows:
        request_id, profile = str(row.get("request_id") or ""), str(row.get("profile") or "")
        if request_id not in observed:
            raise CanonicalSupervisorError(f"replay contains unknown request_id={request_id}")
        if not _is_true(row.get("ok")) or not _is_true(row.get("measured")):
            raise CanonicalSupervisorError(f"replay has failed measurement request_id={request_id} profile={profile}")
        observed[request_id].add(profile)
    for request_id, seen in observed.items():
        missing = sorted(set(profiles) - seen)
        if missing:
            raise CanonicalSupervisorError(f"missing profiles for request_id={request_id}: {missing}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _table_hash(rows: Iterable[dict[str, Any]]) -> str:
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-token-limit", type=int, required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--mode", choices=("bootstrap", "replay", "all"), default="all")
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    root = Path(args.run_root)
    diagnostics = {"probe_id": args.probe_id, "input_token_limit": args.input_token_limit, "mode": args.mode}
    try:
        rows = load_input_rows(Path(args.input), expected_requests=args.expected_requests, max_requests=args.max_requests)
        if args.mode in {"replay", "all"}:
            fixture_path, manifest_path = root / "canonical_fixture.jsonl", root / "canonical_manifest.json"
            if not fixture_path.exists() or not manifest_path.exists():
                raise CanonicalSupervisorError("replay requires a completed canonical bootstrap fixture and manifest")
            fixture = load_input_rows(fixture_path)
            manifest = CanonicalBootstrapManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
            manifest.validate_fixture(fixture)
            replay_files = [root / "replay" / f"{profile}.csv" for profile in LOSSY_PROFILES]
            if not all(path.exists() for path in replay_files):
                raise CanonicalSupervisorError("replay is incomplete; no merged CSV may be written")
            replay_rows = [row for path in replay_files for row in _read_csv(path)]
            ensure_complete_replay(replay_rows, request_ids={str(row["request_id"]) for row in fixture})
        diagnostics["ok"] = True
    except (CanonicalSupervisorError, CanonicalHistoryError, OSError, json.JSONDecodeError) as exc:
        diagnostics.update({"ok": False, "error": str(exc)})
        root.mkdir(parents=True, exist_ok=True)
        (root / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
