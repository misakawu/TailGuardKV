#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import read_jsonl, write_json, write_jsonl


EXPECTED_PROFILES = (
    "full_gpu",
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
KIVI_PROFILES = EXPECTED_PROFILES[1:5]
H2O_PROFILES = EXPECTED_PROFILES[5:]
SENSITIVE_THRESHOLD = 0.05
LOW_RISK_THRESHOLD = 0.01
STRICT_GAP = 0.02
TAILGUARDKV_REPO = Path("/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline")


def candidate_hash(candidate: dict[str, Any]) -> str:
    return hashlib.sha256(
        "\n".join(
            (
                str(candidate["request_id"]),
                str(candidate["task"]),
                str(candidate["prompt"]),
                str(candidate["reference"]),
            )
        ).encode("utf-8")
    ).hexdigest()


def partition_candidates(
    candidates: list[dict[str, Any]], *, output_root: Path, batch_size: int = 60
) -> list[dict[str, Any]]:
    if batch_size != 60:
        raise ValueError("LongBench direct prescreen batches must contain exactly 60 requests")
    manifests: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(candidates), batch_size)):
        batch_candidates = candidates[start : start + batch_size]
        _validate_candidate_order(batch_candidates, start)
        batch_id = f"longbench_prescreen_batch{batch_index:03d}"
        expected_hashes = [_candidate_hash_from_metadata(candidate) for candidate in batch_candidates]
        manifests.append(
            {
                "schema_version": 1,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "request_count": len(batch_candidates),
                "request_ids": [str(candidate["request_id"]) for candidate in batch_candidates],
                "input_hashes": expected_hashes,
                "expected_profiles": list(EXPECTED_PROFILES),
                "candidates_jsonl": str(output_root / f"{batch_id}.jsonl"),
                "csv_output": str(output_root / f"{batch_id}.csv"),
                "manifest_output": str(output_root / f"{batch_id}.manifest.json"),
                "prescreen_output": str(output_root / f"{batch_id}.prescreen.json"),
                "config_output": str(output_root / "configs" / f"{batch_id}.yaml"),
                "log_output": str(output_root / "logs" / f"{batch_id}.log"),
                "pid_output": str(output_root / "pids" / f"{batch_id}.pid"),
                "lock_output": str(output_root / "locks" / "longbench_prescreen_gpu.lock"),
                "candidates": deepcopy(batch_candidates),
            }
        )
    return manifests


def write_partitions(manifests: list[dict[str, Any]]) -> None:
    for manifest in manifests:
        write_jsonl(Path(str(manifest["candidates_jsonl"])), list(manifest["candidates"]))
        write_json(Path(str(manifest["manifest_output"])), manifest)


def write_batch_configs(manifests: list[dict[str, Any]], *, base_config: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to materialize LongBench batch configs") from exc
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"base config must be a mapping: {base_config}")
    for manifest in manifests:
        batch_config = deepcopy(payload)
        data = dict(batch_config.get("data") or {})
        outputs = dict(batch_config.get("outputs") or {})
        data["requests"] = str(manifest["candidates_jsonl"])
        data["max_requests"] = int(manifest["request_count"])
        outputs["smoke_profiles"] = str(manifest["csv_output"])
        batch_config["data"] = data
        batch_config["outputs"] = outputs
        config_path = Path(str(manifest["config_output"]))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(batch_config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_async_serial_launch_command(manifest: dict[str, Any], *, config_path: Path) -> str:
    output = Path(str(manifest["csv_output"]))
    log = Path(str(manifest["log_output"]))
    pid = Path(str(manifest["pid_output"]))
    lock = Path(str(manifest["lock_output"]))
    return (
        "mkdir -p "
        f"'{output.parent}' '{log.parent}' '{pid.parent}' '{lock.parent}'; "
        f"cd '{TAILGUARDKV_REPO}' && "
        f"nohup setsid flock -n '{lock}' env CUDA_VISIBLE_DEVICES=0,1 "
        "conda run -n tailguardkv-base python -m run_util.build_profile_table "
        f"--config '{config_path}' --output '{output}' --no-dry-run "
        f"> '{log}' 2>&1 < /dev/null & echo $! > '{pid}'"
    )


def launch_next_batch(
    manifests: list[dict[str, Any]], *, selected_batch_id: str | None = None
) -> dict[str, Any] | None:
    """Launch at most one GPU-exclusive batch; later batches require a new invocation."""
    for manifest in manifests:
        csv_path = Path(str(manifest["csv_output"]))
        if csv_path.exists():
            write_prescreen_manifest(manifest, merge_batch_measurements(manifest, csv_path))
            continue
        if selected_batch_id is not None and str(manifest["batch_id"]) != selected_batch_id:
            continue
        config_path = Path(str(manifest["config_output"]))
        if not config_path.exists():
            raise RuntimeError(f"missing batch config for {manifest['batch_id']}: {config_path}")
        subprocess.run(build_async_serial_launch_command(manifest, config_path=config_path), shell=True, check=True)
        return manifest
    return None


def merge_batch_measurements(manifest: dict[str, Any], csv_path: Path) -> dict[str, Any]:
    candidates = {str(row["request_id"]): row for row in manifest["candidates"]}
    if len(candidates) != len(manifest["candidates"]):
        raise ValueError("batch manifest contains duplicate request IDs")
    expected_profiles = tuple(str(profile) for profile in manifest["expected_profiles"])
    rows = _read_csv_rows(csv_path)
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, dict[str, dict[str, str]]] = {request_id: {} for request_id in candidates}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        profile = str(row.get("profile", ""))
        if request_id not in candidates:
            raise ValueError(f"unknown request ID: {request_id}")
        if profile not in expected_profiles:
            raise ValueError(f"unexpected profile for {request_id}: {profile}")
        key = (request_id, profile)
        if key in seen:
            raise ValueError(f"duplicate request/profile: {request_id}/{profile}")
        seen.add(key)
        expected_hash = _candidate_hash_from_metadata(candidates[request_id])
        observed_hash = str(row.get("candidate_hash") or "")
        if not observed_hash:
            observed_hash = _measurement_candidate_hash(row, candidates[request_id])
        if observed_hash != expected_hash:
            raise ValueError(f"candidate hash mismatch: {request_id}/{profile}")
        if not _parse_bool(row.get("ok")) or not _parse_bool(row.get("measured")):
            raise ValueError(f"row must have ok=true and measured=true: {request_id}/{profile}")
        grouped[request_id][profile] = row

    for request_id, profile_rows in grouped.items():
        missing = [profile for profile in expected_profiles if profile not in profile_rows]
        if missing:
            raise ValueError(f"missing profiles for {request_id}: {','.join(missing)}")

    ranked = []
    for request_id, profile_rows in grouped.items():
        profile_losses = {profile: _quality_loss(profile_rows[profile], request_id, profile) for profile in expected_profiles}
        risk_family, margin = strict_direct_risk(profile_losses)
        ranked.append(
            {
                "request_id": request_id,
                "candidate_hash": _candidate_hash_from_metadata(candidates[request_id]),
                "direct_risk_family": risk_family,
                "direct_risk_margin": round(margin, 6),
                "direct_profile_losses": profile_losses,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["direct_risk_family"] is None,
            -float(row["direct_risk_margin"]),
            str(row["request_id"]),
        )
    )
    return {
        "schema_version": 1,
        "batch_id": manifest["batch_id"],
        "request_count": manifest["request_count"],
        "expected_profiles": list(expected_profiles),
        "input_hashes": list(manifest["input_hashes"]),
        "csv_output": str(csv_path),
        "ranked_candidates": ranked,
        "candidates": deepcopy(list(manifest["candidates"])),
    }


def write_prescreen_manifest(manifest: dict[str, Any], prescreen_manifest: dict[str, Any]) -> Path:
    output = Path(str(manifest["prescreen_output"]))
    write_json(output, prescreen_manifest)
    return output


def strict_direct_risk(profile_losses: dict[str, float]) -> tuple[str | None, float]:
    kivi_max = max(profile_losses[profile] for profile in KIVI_PROFILES)
    h2o_max = max(profile_losses[profile] for profile in H2O_PROFILES)
    overall_max = max(profile_losses[profile] for profile in EXPECTED_PROFILES if profile != "full_gpu")
    if overall_max <= LOW_RISK_THRESHOLD:
        return "low_risk", LOW_RISK_THRESHOLD - overall_max
    if kivi_max >= SENSITIVE_THRESHOLD and kivi_max - h2o_max >= STRICT_GAP:
        return "kivi_sensitive", kivi_max - h2o_max
    if h2o_max >= SENSITIVE_THRESHOLD and h2o_max - kivi_max >= STRICT_GAP:
        return "h2o_sensitive", h2o_max - kivi_max
    return None, 0.0


def prescreen_candidates_for_hybrid(prescreen_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {str(row["request_id"]): row for row in prescreen_manifest["candidates"]}
    selected: list[dict[str, Any]] = []
    for rank, ranked in enumerate(prescreen_manifest["ranked_candidates"]):
        candidate = deepcopy(candidates[str(ranked["request_id"])])
        candidate.pop("direct_risk_family", None)
        metadata = dict(candidate.get("metadata") or {})
        metadata.pop("direct_risk_family", None)
        metadata.pop("direct_risk_margin", None)
        metadata["prescreen_rank"] = rank
        metadata["prescreen_batch_id"] = prescreen_manifest["batch_id"]
        candidate["metadata"] = metadata
        selected.append(candidate)
    return selected


def next_batch_for_hybrid_need(
    manifests: list[dict[str, Any]],
    completed_prescreens: list[dict[str, Any]],
    *,
    required_per_cell: int = 12,
) -> dict[str, Any] | None:
    """Return the next LongBench batch until every strict risk/task pool is covered."""
    if required_per_cell <= 0:
        raise ValueError("required_per_cell must be positive")
    counts = {
        (risk_family, task): 0
        for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk")
        for task in ("qa", "summary")
    }
    completed_ids: set[str] = set()
    for prescreen in completed_prescreens:
        completed_ids.add(str(prescreen["batch_id"]))
        tasks = {str(candidate["request_id"]): str(candidate["task"]) for candidate in prescreen["candidates"]}
        for ranked in prescreen["ranked_candidates"]:
            risk_family = ranked.get("direct_risk_family")
            task = tasks.get(str(ranked["request_id"]))
            key = (risk_family, task)
            if key in counts:
                counts[key] += 1
    if all(count >= required_per_cell for count in counts.values()):
        return None
    return next((manifest for manifest in manifests if str(manifest["batch_id"]) not in completed_ids), None)


def hybrid_need_status(
    manifests: list[dict[str, Any]],
    completed_prescreens: list[dict[str, Any]],
    *,
    required_per_cell: int,
) -> dict[str, Any]:
    next_batch = next_batch_for_hybrid_need(
        manifests, completed_prescreens, required_per_cell=required_per_cell
    )
    counts = _hybrid_need_counts(completed_prescreens)
    need_met = all(count >= required_per_cell for count in counts.values())
    if need_met:
        status = "need_met"
    elif next_batch is None:
        status = "longbench_exhausted"
    else:
        status = "pending"
    return {
        "status": status,
        "required_per_cell": required_per_cell,
        "counts": {f"{risk}/{task}": count for (risk, task), count in sorted(counts.items())},
        "next_batch_id": next_batch["batch_id"] if next_batch is not None else None,
    }


def _validate_candidate_order(candidates: list[dict[str, Any]], start: int) -> None:
    for offset, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") or {}
        if int(metadata.get("candidate_order", -1)) != start + offset:
            raise ValueError("candidates must retain contiguous global candidate_order")
        _candidate_hash_from_metadata(candidate)


def _candidate_hash_from_metadata(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    value = str(metadata.get("candidate_hash") or "")
    expected = candidate_hash(candidate)
    if value != expected:
        raise ValueError(f"candidate_hash mismatch for {candidate.get('request_id', '')}")
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"request_id", "profile", "ok", "measured", "quality_loss"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"measurement CSV missing columns: {','.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _measurement_candidate_hash(row: dict[str, str], candidate: dict[str, Any]) -> str:
    if str(row.get("task") or "") != str(candidate["task"]):
        raise ValueError(f"task mismatch: {row.get('request_id', '')}")
    prompt = row.get("extra_prompt_text")
    reference = row.get("extra_reference")
    if prompt is None or reference is None:
        return ""
    measured_candidate = {
        "request_id": row.get("request_id", ""),
        "task": candidate["task"],
        "prompt": prompt,
        "reference": reference,
    }
    return candidate_hash(measured_candidate)


def _quality_loss(row: dict[str, str], request_id: str, profile: str) -> float:
    try:
        return float(row["quality_loss"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid quality_loss: {request_id}/{profile}") from exc


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _hybrid_need_counts(completed_prescreens: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts = {
        (risk_family, task): 0
        for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk")
        for task in ("qa", "summary")
    }
    for prescreen in completed_prescreens:
        tasks = {str(candidate["request_id"]): str(candidate["task"]) for candidate in prescreen["candidates"]}
        for ranked in prescreen["ranked_candidates"]:
            key = (ranked.get("direct_risk_family"), tasks.get(str(ranked["request_id"])))
            if key in counts:
                counts[key] += 1
    return counts


def load_completed_prescreens(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.prescreen.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and serially launch LongBench direct-prescreen batches.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config")
    parser.add_argument("--completed-prescreens-dir")
    parser.add_argument("--required-per-cell", type=int, default=12)
    parser.add_argument("--launch-next", action="store_true")
    args = parser.parse_args()
    manifests = partition_candidates(read_jsonl(Path(args.candidates)), output_root=Path(args.output_root))
    write_partitions(manifests)
    if args.base_config:
        write_batch_configs(manifests, base_config=Path(args.base_config))
    completed = (
        load_completed_prescreens(Path(args.completed_prescreens_dir))
        if args.completed_prescreens_dir
        else []
    )
    status = hybrid_need_status(manifests, completed, required_per_cell=args.required_per_cell)
    if args.launch_next:
        if not args.base_config:
            raise SystemExit("--launch-next requires --base-config")
        launched = (
            launch_next_batch(manifests, selected_batch_id=str(status["next_batch_id"]))
            if status["status"] == "pending"
            else None
        )
        print(json.dumps({**status, "launched": launched["batch_id"] if launched else None}, ensure_ascii=False))
    else:
        print(json.dumps({**status, "batches": len(manifests)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
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
LOW_RISK_TASK_FLOOR = 20


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
    raise RuntimeError(
        "direct prescreen measurements cannot create a final quality fixture; "
        "Task 5 must use fully shaped final-input measurements instead"
    )
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in measurements:
        if row.quality_loss is None or row.profile == "full_gpu":
            continue
        grouped[row.request_id][row.profile] = max(float(row.quality_loss), grouped[row.request_id].get(row.profile, 0.0))

    labeled: list[dict[str, Any]] = []
    tie_sensitive_rows: list[dict[str, Any]] = []
    for request_id, profile_losses in grouped.items():
        candidate = candidates.get(request_id)
        if candidate is None:
            continue
        risk_family = classify_profile_losses(profile_losses)
        if risk_family is None:
            continue
        row = {
            **candidate,
            "metadata": {
                **candidate["metadata"],
                "risk_family": risk_family,
                "kivi_max_loss": round(_family_max(profile_losses, MAINSTREAM_KIVI_PROFILES), 6),
                "h2o_max_loss": round(_family_max(profile_losses, MAINSTREAM_H2O_PROFILES), 6),
            },
        }
        if risk_family == "tie_sensitive":
            tie_sensitive_rows.append(row)
        else:
            labeled.append(row)

    selected = select_balanced_quality_rows(labeled + tie_sensitive_rows)
    for index, row in enumerate(selected):
        row["metadata"]["split"] = "calibration" if index % 2 == 0 else "eval"

    manifest = {
        "rows": len(selected),
        "risk_distribution": _count_by(selected, "risk_family"),
        "task_distribution": _count_task(selected),
        "splits": _count_split(selected),
        "source_candidates": len(candidates),
        "labeled_candidates": len(labeled) + len(tie_sensitive_rows),
        "strict_risk_distribution": _count_by(labeled, "risk_family"),
        "tie_sensitive_candidates": len(tie_sensitive_rows),
    }
    return selected, manifest


def classify_profile_losses(profile_losses: dict[str, float]) -> str | None:
    kivi_max = _family_max(profile_losses, MAINSTREAM_KIVI_PROFILES)
    h2o_max = _family_max(profile_losses, MAINSTREAM_H2O_PROFILES)
    overall_max = max([*profile_losses.values(), 0.0])
    if overall_max <= LOW_RISK_THRESHOLD:
        return "low_risk"
    kivi_sensitive = kivi_max >= SENSITIVE_THRESHOLD
    h2o_sensitive = h2o_max >= SENSITIVE_THRESHOLD
    if kivi_sensitive and not h2o_sensitive:
        return "kivi_sensitive"
    if h2o_sensitive and not kivi_sensitive:
        return "h2o_sensitive"
    if kivi_sensitive and h2o_sensitive:
        if abs(kivi_max - h2o_max) <= TIE_EPSILON:
            return "tie_sensitive"
        return "kivi_sensitive" if kivi_max > h2o_max else "h2o_sensitive"
    return None


def select_balanced_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metadata"]["risk_family"])].append(row)
    targets = compute_selection_targets(
        strict_counts={
            "kivi_sensitive": len(grouped.get("kivi_sensitive", [])),
            "h2o_sensitive": len(grouped.get("h2o_sensitive", [])),
            "low_risk": len(grouped.get("low_risk", [])),
        },
        tie_sensitive_count=len(grouped.get("tie_sensitive", [])),
    )
    selected: list[dict[str, Any]] = []
    for risk_family, target in targets.items():
        ranked = sorted(
            grouped.get(risk_family, []),
            key=lambda row: (len(str(row["prompt"])), str(row["metadata"].get("source_dataset", "")), str(row["request_id"])),
        )
        if risk_family != "low_risk" and len(ranked) < target:
            tie_ranked = sorted(
                grouped.get("tie_sensitive", []),
                key=lambda row: (
                    abs(float(row["metadata"].get("kivi_max_loss", 0.0)) - float(row["metadata"].get("h2o_max_loss", 0.0))),
                    len(str(row["prompt"])),
                    str(row["request_id"]),
                ),
            )
            backfill: list[dict[str, Any]] = []
            for row in tie_ranked[: max(0, target - len(ranked))]:
                copied = deepcopy(row)
                copied["metadata"]["risk_family"] = risk_family
                copied["metadata"]["tie_backfill"] = True
                backfill.append(copied)
            ranked = ranked + backfill
        if len(ranked) < target:
            raise RuntimeError(f"{risk_family} 样本不足: {len(ranked)} < {target}")
        selected.extend(ranked[:target])
    low_risk_rows = grouped.get("low_risk", [])
    selected_low_risk = _select_low_risk_rows(low_risk_rows, targets["low_risk"])
    selected = [row for row in selected if str(row["metadata"]["risk_family"]) != "low_risk"] + selected_low_risk
    return sorted(selected, key=lambda row: (str(row["metadata"]["risk_family"]), str(row["task"]), len(str(row["prompt"]))))


def compute_selection_targets(
    *,
    strict_counts: dict[str, int],
    tie_sensitive_count: int,
) -> dict[str, int]:
    kivi_count = int(strict_counts.get("kivi_sensitive", 0))
    h2o_count = int(strict_counts.get("h2o_sensitive", 0))
    low_risk_count = int(strict_counts.get("low_risk", 0))
    base_sensitive = min(TARGETS["kivi_sensitive"], TARGETS["h2o_sensitive"], max(kivi_count, h2o_count) + (tie_sensitive_count // 2))
    if kivi_count == h2o_count:
        kivi_target = base_sensitive
        h2o_target = base_sensitive
    else:
        smaller = min(kivi_count, h2o_count)
        larger = max(kivi_count, h2o_count)
        boosted = min(base_sensitive, smaller + tie_sensitive_count)
        if kivi_count < h2o_count:
            kivi_target, h2o_target = boosted, min(base_sensitive, larger)
        else:
            kivi_target, h2o_target = min(base_sensitive, larger), boosted
    return {
        "kivi_sensitive": kivi_target,
        "h2o_sensitive": h2o_target,
        "low_risk": min(TARGETS["low_risk"], low_risk_count),
    }


def _select_low_risk_rows(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    selected: list[dict[str, Any]] = []
    for task, task_rows in sorted(by_task.items()):
        ranked = sorted(task_rows, key=lambda row: (len(str(row["prompt"])), str(row["request_id"])))
        floor = min(LOW_RISK_TASK_FLOOR, len(ranked), target - len(selected))
        selected.extend(ranked[:floor])
    if len(selected) >= target:
        return selected[:target]
    remaining = [row for row in sorted(rows, key=lambda row: (len(str(row["prompt"])), str(row["request_id"]))) if row not in selected]
    selected.extend(remaining[: max(0, target - len(selected))])
    return selected[:target]


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


def _count_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get("split", ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESCREEN_ROOT="${PRESCREEN_ROOT:-$ROOT_DIR/artifacts/prescreen/longbench}"
COMPLETED_PRESCREENS_DIR="${COMPLETED_PRESCREENS_DIR:-$PRESCREEN_ROOT}"
REQUIRED_PER_CELL="${REQUIRED_PER_CELL:-12}"

mkdir -p "$ROOT_DIR/artifacts/tmp"

python "$ROOT_DIR/scripts/prepare_longbench_candidates.py"
python "$ROOT_DIR/scripts/partition_labeling_batches.py" \
  --candidates "$ROOT_DIR/artifacts/candidates/longbench_candidates.jsonl" \
  --output-root "$PRESCREEN_ROOT" \
  --base-config "$ROOT_DIR/configs/longbench_labeling.yaml" \
  --completed-prescreens-dir "$COMPLETED_PRESCREENS_DIR" \
  --required-per-cell "$REQUIRED_PER_CELL" \
  --launch-next

echo "longbench direct prescreen scheduled or stopped at its capacity gate"
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from partition_labeling_batches import (
    EXPECTED_PROFILES,
    build_async_serial_launch_command,
    merge_batch_measurements,
    next_batch_for_hybrid_need,
    partition_candidates,
    prescreen_candidates_for_hybrid,
    write_prescreen_manifest,
    write_batch_configs,
)
from label_longbench_quality import build_fixture


def _candidate(index: int) -> dict[str, object]:
    prompt = f"prompt {index}"
    reference = f"reference {index}"
    request_id = f"longbench_qa_{index:05d}"
    candidate_hash = hashlib.sha256(
        f"{request_id}\nqa\n{prompt}\n{reference}".encode("utf-8")
    ).hexdigest()
    return {
        "request_id": request_id,
        "task": "qa",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "candidate_order": index,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "candidate_hash": candidate_hash,
            "risk_family": "unlabeled",
        },
    }


def _measurement(candidate: dict[str, object], profile: str, *, candidate_hash: str | None = None) -> dict[str, str]:
    metadata = candidate["metadata"]
    assert isinstance(metadata, dict)
    return {
        "request_id": str(candidate["request_id"]),
        "profile": profile,
        "candidate_hash": candidate_hash or str(metadata["candidate_hash"]),
        "ok": "true",
        "measured": "true",
        "quality_loss": "0.01",
    }


def test_identical_candidates_produce_stable_sixty_request_manifests() -> None:
    candidates = [_candidate(index) for index in range(61)]

    first = partition_candidates(candidates, output_root=Path("/tmp/prescreen"))
    second = partition_candidates(candidates, output_root=Path("/tmp/prescreen"))

    assert first == second
    assert [manifest["request_count"] for manifest in first] == [60, 1]
    assert first[0]["expected_profiles"] == list(EXPECTED_PROFILES)
    assert first[0]["input_hashes"] == [
        candidates[index]["metadata"]["candidate_hash"] for index in range(60)
    ]
    assert first[0]["csv_output"].endswith("longbench_prescreen_batch000.csv")
    command = build_async_serial_launch_command(first[0], config_path=Path("configs/longbench_labeling.yaml"))
    assert "cd '/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline' &&" in command
    assert "nohup setsid flock -n" in command
    assert "env CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base" in command
    assert "--no-dry-run" in command
    assert "longbench_prescreen_batch000.pid" in command


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        ("duplicate", "duplicate request/profile"),
        ("hash", "candidate hash mismatch"),
        ("failed", "ok=true and measured=true"),
        ("incomplete", "missing profiles"),
    ],
)
def test_merger_rejects_unverifiable_measurement_rows(tmp_path: Path, mutate: str, error: str) -> None:
    candidate = _candidate(0)
    manifest = partition_candidates([candidate], output_root=tmp_path)[0]
    rows = [_measurement(candidate, profile) for profile in EXPECTED_PROFILES]
    if mutate == "duplicate":
        rows.append(_measurement(candidate, EXPECTED_PROFILES[0]))
    elif mutate == "hash":
        rows[0]["candidate_hash"] = "0" * 64
    elif mutate == "failed":
        rows[0]["ok"] = "false"
    elif mutate == "incomplete":
        rows.pop()

    csv_path = tmp_path / "measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=error):
        merge_batch_measurements(manifest, csv_path)


def test_direct_risk_stays_in_prescreen_manifest_not_hybrid_candidate_risk_fields(tmp_path: Path) -> None:
    candidate = _candidate(0)
    manifest = partition_candidates([candidate], output_root=tmp_path)[0]
    rows = [_measurement(candidate, profile) for profile in EXPECTED_PROFILES]
    for row in rows:
        row["quality_loss"] = "0.07" if row["profile"].startswith("kivi") else "0.01"
    csv_path = tmp_path / "measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    prescreen = merge_batch_measurements(manifest, csv_path)
    prescreen_path = write_prescreen_manifest(manifest, prescreen)
    hybrid_candidates = prescreen_candidates_for_hybrid(prescreen)

    assert prescreen["ranked_candidates"][0]["direct_risk_family"] == "kivi_sensitive"
    assert json.loads(prescreen_path.read_text(encoding="utf-8"))["ranked_candidates"][0]["direct_risk_family"] == "kivi_sensitive"
    assert hybrid_candidates[0]["metadata"]["risk_family"] == "unlabeled"
    assert "direct_risk_family" not in hybrid_candidates[0]
    assert "direct_risk_family" not in hybrid_candidates[0]["metadata"]


def test_direct_prescreen_measurements_cannot_export_a_final_quality_fixture() -> None:
    with pytest.raises(RuntimeError, match="direct prescreen"):
        build_fixture({}, [])


def test_batch_configs_bind_each_runner_invocation_to_its_own_input_and_csv(tmp_path: Path) -> None:
    manifests = partition_candidates([_candidate(index) for index in range(61)], output_root=tmp_path / "batches")
    base_config = tmp_path / "longbench.yaml"
    base_config.write_text(
        "data:\n  requests: original.jsonl\n  max_requests: 60\noutputs:\n  smoke_profiles: original.csv\n",
        encoding="utf-8",
    )

    write_batch_configs(manifests, base_config=base_config)

    first_config = Path(str(manifests[0]["config_output"])).read_text(encoding="utf-8")
    second_config = Path(str(manifests[1]["config_output"])).read_text(encoding="utf-8")
    assert str(manifests[0]["candidates_jsonl"]) in first_config
    assert str(manifests[0]["csv_output"]) in first_config
    assert str(manifests[1]["candidates_jsonl"]) in second_config
    assert str(manifests[1]["csv_output"]) in second_config


def test_scheduler_advances_only_until_strict_hybrid_prescreen_need_is_met(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(120)]
    candidates[60]["task"] = "summary"
    candidates[60]["metadata"]["candidate_hash"] = hashlib.sha256(
        f"{candidates[60]['request_id']}\nsummary\n{candidates[60]['prompt']}\n{candidates[60]['reference']}".encode("utf-8")
    ).hexdigest()
    manifests = partition_candidates(candidates, output_root=tmp_path)
    completed = {
        "batch_id": manifests[0]["batch_id"],
        "candidates": manifests[0]["candidates"],
        "ranked_candidates": [
            {
                "request_id": candidate["request_id"],
                "direct_risk_family": ("kivi_sensitive", "h2o_sensitive", "low_risk")[index % 3],
            }
            for index, candidate in enumerate(manifests[0]["candidates"])
        ],
    }

    assert next_batch_for_hybrid_need(manifests, [completed], required_per_cell=1) == manifests[1]

    complete = {
        "batch_id": manifests[1]["batch_id"],
        "candidates": [
            {**candidate, "task": "summary"}
            for candidate in manifests[1]["candidates"]
        ],
        "ranked_candidates": [
            {
                "request_id": candidate["request_id"],
                "direct_risk_family": ("kivi_sensitive", "h2o_sensitive", "low_risk")[index % 3],
            }
            for index, candidate in enumerate(manifests[1]["candidates"])
        ],
    }
    assert next_batch_for_hybrid_need(manifests, [completed, complete], required_per_cell=1) is None


def test_merger_rejects_task_mismatch_when_csv_uses_prompt_reference_hash_fallback(tmp_path: Path) -> None:
    candidate = _candidate(0)
    manifest = partition_candidates([candidate], output_root=tmp_path)[0]
    rows = [
        {
            "request_id": str(candidate["request_id"]),
            "profile": profile,
            "task": "summary",
            "extra_prompt_text": str(candidate["prompt"]),
            "extra_reference": str(candidate["reference"]),
            "ok": "true",
            "measured": "true",
            "quality_loss": "0.01",
        }
        for profile in EXPECTED_PROFILES
    ]
    csv_path = tmp_path / "measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="task mismatch"):
        merge_batch_measurements(manifest, csv_path)


def test_cli_loads_completed_prescreens_and_stops_when_every_cell_is_sufficient(tmp_path: Path) -> None:
    risks = ("kivi_sensitive", "h2o_sensitive", "low_risk")
    candidates = [_candidate(index) for index in range(6)]
    for index, candidate in enumerate(candidates):
        task = "qa" if index < 3 else "summary"
        candidate["task"] = task
        candidate["metadata"]["candidate_hash"] = hashlib.sha256(
            f"{candidate['request_id']}\n{task}\n{candidate['prompt']}\n{candidate['reference']}".encode("utf-8")
        ).hexdigest()
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text("\n".join(json.dumps(candidate) for candidate in candidates) + "\n", encoding="utf-8")
    output_root = tmp_path / "batches"
    manifest = partition_candidates(candidates, output_root=output_root)[0]
    completed_dir = tmp_path / "completed"
    completed_dir.mkdir()
    (completed_dir / "complete.prescreen.json").write_text(
        json.dumps(
            {
                "batch_id": manifest["batch_id"],
                "candidates": candidates,
                "ranked_candidates": [
                    {"request_id": candidate["request_id"], "direct_risk_family": risks[index % 3]}
                    for index, candidate in enumerate(candidates)
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "partition_labeling_batches.py"),
            "--candidates",
            str(candidates_path),
            "--output-root",
            str(output_root),
            "--completed-prescreens-dir",
            str(completed_dir),
            "--required-per-cell",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["status"] == "need_met"
