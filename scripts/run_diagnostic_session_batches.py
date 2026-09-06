#!/usr/bin/env python3
"""Run diagnostic session smoke in isolated complete-session batches."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass(frozen=True)
class SessionBatch:
    batch_id: str
    fixture_path: Path
    session_count: int
    request_count: int


_PRESSURE_TRACE_REQUEST_ID = re.compile(r"__pressure_r\d+_c\d+$")


def _canonical_request_id(request_id: str) -> str:
    """Map the runtime's pressure-trace request suffix back to its fixture ID."""
    return _PRESSURE_TRACE_REQUEST_ID.sub("", request_id)


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def is_diagnostic_gate_failure(run_dir: Path) -> bool:
    """Return whether the current batch summary reports only a diagnostic gate failure."""
    gate_dir = run_dir / "profile_tables"
    failed_gate = False
    for path in gate_dir.glob("*_gate.json"):
        if "trace" not in path.name and "risk" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("passed") is False:
            failed_gate = True
    if not failed_gate:
        return False

    summary_paths = sorted(
        path
        for path in (run_dir / "policy_tables").glob("*_summary.csv")
        if not path.name.endswith("_total_summary.csv")
    )
    if len(summary_paths) != 1:
        return False
    try:
        with summary_paths[0].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    errors = [str(row.get("error") or "").strip().casefold() for row in rows if str(row.get("error") or "").strip()]
    diagnostic_gate_errors = {
        "session trace semantics gate failed",
        "session risk signal gate failed",
        "trace semantics gate failed",
        "risk signal gate failed",
    }
    return len(errors) == 1 and errors[0] in diagnostic_gate_errors


def validate_batch_output(batch: SessionBatch, run_dir: Path, profile_names: set[str]) -> dict[str, Any]:
    """Validate complete successful profile coverage for one materialized batch."""
    errors: list[str] = []
    fixture_ids: list[str] = []
    try:
        fixture_rows = [json.loads(line) for line in batch.fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in fixture_rows:
            request_id = str(row.get("request_id") or "")
            if not request_id:
                errors.append("empty fixture request_id")
            elif request_id in fixture_ids:
                errors.append(f"duplicate fixture request_id: {request_id}")
            fixture_ids.append(request_id)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        errors.append(f"invalid batch fixture: {exc}")

    if len(fixture_ids) != batch.request_count:
        errors.append(
            f"fixture request count mismatch: expected {batch.request_count}, found {len(fixture_ids)}"
        )

    csv_paths = sorted((run_dir / "profile_tables").glob("*_profiles.csv"))
    rows: list[dict[str, Any]] = []
    if len(csv_paths) != 1:
        errors.append(f"expected one profile CSV, found {len(csv_paths)}")
    else:
        try:
            with csv_paths[0].open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required_columns = {"request_id", "profile", "ok", "measured"}
                if not required_columns.issubset(set(reader.fieldnames or ())):
                    errors.append(f"invalid profile CSV: missing columns {sorted(required_columns - set(reader.fieldnames or ()))}")
                else:
                    rows = list(reader)
        except (OSError, csv.Error) as exc:
            errors.append(f"invalid profile CSV: {exc}")

    expected = {(request_id, profile) for request_id in fixture_ids for profile in profile_names}
    observed: set[tuple[str, str]] = set()
    for row in rows:
        key = (
            _canonical_request_id(str(row.get("request_id") or "")),
            str(row.get("profile") or ""),
        )
        if key in observed:
            errors.append(f"duplicate profile coverage: request_id={key[0]} profile={key[1]}")
        observed.add(key)
        if not _is_true(row.get("ok")) or not _is_true(row.get("measured")):
            errors.append(f"failed measurement: request_id={key[0]} profile={key[1]}")
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        errors.append(f"unknown profile coverage: {unknown}")
    if missing:
        errors.append(f"missing profile coverage: {missing}")
    return {
        "batch_id": batch.batch_id,
        "mergeable": not errors,
        "profile_rows": len(rows),
        "expected_profile_rows": len(expected),
        "errors": errors,
    }


def materialize_session_batches(fixture_path: Path, output_root: Path, *, sessions_per_batch: int) -> list[SessionBatch]:
    if sessions_per_batch <= 0:
        raise ValueError("sessions_per_batch must be positive")
    rows = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session_id = str(row.get("session_id") or "")
        if not session_id:
            raise ValueError("diagnostic fixture has a row without session_id")
        by_session.setdefault(session_id, []).append(row)
    session_ids = sorted(by_session, key=lambda session_id: min(int(row["arrival_index"]) for row in by_session[session_id]))
    output_root.mkdir(parents=True, exist_ok=True)
    batches: list[SessionBatch] = []
    for index, start in enumerate(range(0, len(session_ids), sessions_per_batch)):
        selected = set(session_ids[start : start + sessions_per_batch])
        batch_rows = [dict(row) for row in rows if str(row["session_id"]) in selected]
        batch_rows.sort(key=lambda row: int(row["arrival_index"]))
        for arrival_index, row in enumerate(batch_rows):
            row["arrival_index"] = arrival_index
            row["metadata"] = {**dict(row.get("metadata") or {}), "diagnostic_only": True, "batch_id": f"batch{index:03d}"}
        path = output_root / f"batch{index:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch_rows), encoding="utf-8")
        batches.append(SessionBatch(f"batch{index:03d}", path, len(selected), len(batch_rows)))
    return batches


def _write_batch_config(base_config: dict[str, Any], batch: SessionBatch, config_path: Path, run_dir: Path) -> None:
    config = dict(base_config)
    config["diagnostic_only"] = True
    data = dict(config.get("data") or {})
    data.update({"requests": str(batch.fixture_path.resolve()), "max_requests": batch.request_count, "diagnostic_only": True})
    config["data"] = data
    config["session_trace"] = {"copies": 1, "repeat_rounds": 1, "memory_budgets_mib": [4900]}
    outputs = dict(config.get("outputs") or {})
    config["outputs"] = {
        name: (
            str(Path(Path(path).parent.name) / Path(path).name)
            if Path(path).parent.name
            else Path(path).name
        )
        for name, path in outputs.items()
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _batch_from_manifest_item(item: dict[str, Any]) -> SessionBatch:
    return SessionBatch(
        str(item["batch_id"]),
        Path(item["fixture"]),
        int(item["sessions"]),
        int(item["requests"]),
    )


def _profile_names(manifest: dict[str, Any], item: dict[str, Any]) -> set[str]:
    names = item.get("profile_names", manifest.get("profile_names"))
    if names is None:
        config = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8")) or {}
        names = (config.get("profiles") or {}).get("names") or []
    return {str(name) for name in names}


def _artifact_path(run_dir: Path, directory: str, pattern: str) -> str | None:
    paths = sorted((run_dir / directory).glob(pattern))
    return str(paths[0]) if len(paths) == 1 else None


def _remove_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"expected output directory: {path}")
        shutil.rmtree(path)


def _write_supervisor_manifest(
    root: Path,
    manifest: dict[str, Any],
    statuses: list[dict[str, Any]],
    *,
    merged: bool,
    merge_error: str | None = None,
) -> None:
    payload = {
        "diagnostic_only": True,
        "merged": merged,
        "profile_names": sorted(str(name) for name in manifest.get("profile_names", [])),
        "batches": statuses,
    }
    if merge_error is not None:
        payload["merge_error"] = merge_error
    (root / "supervisor_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames or any(not field for field in fieldnames) or len(set(fieldnames)) != len(fieldnames):
                raise ValueError(f"invalid CSV header: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"invalid CSV: {path}: {exc}") from exc
    for row in rows:
        if set(row) != set(fieldnames) or None in row or any(value is None for value in row.values()):
            raise ValueError(f"invalid CSV row shape: {path}")
    return fieldnames, rows


def _merge_csv_files(source_paths: list[Path], target_path: Path) -> None:
    expected_fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    for source in source_paths:
        fieldnames, source_rows = _read_csv_rows(source)
        if expected_fields is None:
            expected_fields = fieldnames
        elif fieldnames != expected_fields:
            raise ValueError(f"CSV header mismatch: {source}")
        rows.extend(source_rows)
    if expected_fields is None:
        raise ValueError("no CSV sources to merge")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=expected_fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_batch_outputs(statuses: list[dict[str, Any]], root: Path) -> Path:
    """Merge already-validated diagnostic batch artifacts in manifest order."""
    merged_root = root / "merged"
    staging_root = root / ".merged.tmp"
    _remove_output_directory(staging_root)
    if merged_root.exists():
        raise ValueError(f"merge destination already exists: {merged_root}")
    profile_paths = [Path(str(status["profile_csv"])) for status in statuses]
    try:
        _merge_csv_files(profile_paths, staging_root / "profile_tables" / profile_paths[0].name)
        trace_paths = [status.get("trace_csv") for status in statuses]
        if all(trace_paths):
            source_paths = [Path(str(path)) for path in trace_paths]
            _merge_csv_files(source_paths, staging_root / "session_traces" / source_paths[0].name)
        merged_manifest = {
            "diagnostic_only": True,
            "merged": True,
            "batches": statuses,
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return staging_root.replace(merged_root)
    except Exception:
        _remove_output_directory(staging_root)
        raise


def run_batches(
    manifest: dict[str, Any],
    root: Path,
    repo_root: Path,
    conda_env: str,
    runner: Callable[[dict[str, Any]], int],
) -> int:
    """Run every diagnostic batch serially and merge only fully valid outputs."""
    del repo_root, conda_env  # Kept in the public interface for child-runner construction.
    root.mkdir(parents=True, exist_ok=True)
    _remove_output_directory(root / "merged")
    statuses: list[dict[str, Any]] = []
    for raw_item in manifest.get("batches", []):
        item = dict(raw_item)
        batch = _batch_from_manifest_item(item)
        run_dir = Path(item.get("run_dir", root / "batch_outputs" / batch.batch_id))
        execution_error: str | None = None
        try:
            _remove_output_directory(run_dir)
            returncode: int | None = runner(item)
        except Exception as exc:
            returncode = None
            execution_error = f"{type(exc).__name__}: {exc}"
        profile_names = _profile_names(manifest, item)
        status = validate_batch_output(batch, run_dir, profile_names)
        diagnostic_gate_failed = is_diagnostic_gate_failure(run_dir)
        gate_only_failure = returncode is not None and returncode != 0 and diagnostic_gate_failed
        status.update(
            {
                "diagnostic_only": True,
                "fixture": str(batch.fixture_path),
                "config": str(item["config"]),
                "run_dir": str(run_dir),
                "returncode": returncode,
                "expected_requests": batch.request_count,
                "expected_profiles": sorted(profile_names),
                "profile_csv": _artifact_path(run_dir, "profile_tables", "*_profiles.csv"),
                "trace_csv": _artifact_path(run_dir, "session_traces", "*_trace.csv"),
                "gate_only_failure": gate_only_failure,
                "diagnostic_gate_failed": diagnostic_gate_failed,
            }
        )
        if execution_error is not None:
            status["execution_error"] = execution_error
            status["mergeable"] = False
            status["errors"].append(f"runner exception: {execution_error}")
        elif returncode != 0 and not (status["mergeable"] and gate_only_failure):
            status["mergeable"] = False
            status["errors"].append(f"child exited with return code {returncode}")
        statuses.append(status)
        _write_supervisor_manifest(root, manifest, statuses, merged=False)

    mergeable = bool(statuses) and all(status["mergeable"] for status in statuses)
    if not mergeable:
        _write_supervisor_manifest(root, manifest, statuses, merged=False)
        return 1
    try:
        merge_batch_outputs(statuses, root)
    except Exception as exc:
        _write_supervisor_manifest(root, manifest, statuses, merged=False, merge_error=f"{type(exc).__name__}: {exc}")
        return 1
    _write_supervisor_manifest(root, manifest, statuses, merged=True)
    return 0


def _profile_output_name(item: dict[str, Any]) -> str:
    """Resolve the batch's profile CSV basename from its written batch config."""
    try:
        config = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8")) or {}
        outputs = config.get("outputs") or {}
        return Path(str(outputs.get("smoke_profiles") or "diagnostic_session27_profiles.csv")).name
    except (OSError, yaml.YAMLError):
        return "diagnostic_session27_profiles.csv"


def child_command(repo_root: Path, args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build the batch child command; --profile-only measures profiles without policy runs."""
    python_command = ["python", "run_experiment.py", "pilot-smoke-measured", "--config", item["config"], "--run-dir", item["run_dir"]]
    if args.profile_only:
        run_dir = Path(item["run_dir"])
        profile_output = run_dir / "profile_tables" / _profile_output_name(item)
        python_command = [
            "python",
            "-m",
            "run_util.build_profile_table",
            "--config",
            item["config"],
            "--output",
            str(profile_output),
            "--no-dry-run",
        ]
    return ["conda", "run", "--no-capture-output", "--cwd", str(repo_root), "-n", args.conda_env, *python_command]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sessions-per-batch", type=int, default=3)
    parser.add_argument("--conda-env", default="tailguardkv-base")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="只测量 profiles，不跑 policy 实验轨")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    batches = materialize_session_batches(Path(args.fixture), root / "fixtures", sessions_per_batch=args.sessions_per_batch)
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    profile_names = [str(name) for name in ((base.get("profiles") or {}).get("names") or [])]
    manifest = {"diagnostic_only": True, "sessions_per_batch": args.sessions_per_batch, "profile_names": profile_names, "batches": []}
    for batch in batches:
        config_path = root / "configs" / f"{batch.batch_id}.yaml"
        run_dir = root / "batch_outputs" / batch.batch_id
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_batch_config(base, batch, config_path, run_dir)
        manifest["batches"].append({"batch_id": batch.batch_id, "fixture": str(batch.fixture_path), "config": str(config_path), "run_dir": str(run_dir), "sessions": batch.session_count, "requests": batch.request_count})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    repo_root = Path(__file__).resolve().parent.parent

    def run_child(item: dict[str, Any]) -> int:
        completed = subprocess.run(child_command(repo_root, args, item), cwd=repo_root)
        return completed.returncode

    return run_batches(manifest, root, repo_root, args.conda_env, run_child)


if __name__ == "__main__":
    raise SystemExit(main())
