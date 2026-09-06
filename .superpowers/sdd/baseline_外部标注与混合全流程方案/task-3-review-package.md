#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from common import (
    BASELINE_QUALITY_MANIFEST,
    LONG_BENCH_CANDIDATES,
    LONG_BENCH_ZIP,
    ensure_artifact_dirs,
    ensure_repo_import_path,
    length_bucket,
    normalize_text,
    write_json,
    write_jsonl,
)

ensure_repo_import_path()

from env_asset_prepare.prepare_pilot_assets import format_longbench_prompt


DATASET_GROUPS = {
    "qa": (
        "qasper",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "multifieldqa_en",
    ),
    "summary": (
        "gov_report",
        "qmsum",
        "multi_news",
        "samsum",
        "vcsum",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LongBench QA/Summary/Code candidates for external labeling.")
    parser.add_argument("--output", default=str(LONG_BENCH_CANDIDATES))
    parser.add_argument("--manifest", default=str(BASELINE_QUALITY_MANIFEST))
    parser.add_argument("--max-prompt-chars", type=int, default=3000)
    parser.add_argument("--zip-path", default=str(LONG_BENCH_ZIP))
    args = parser.parse_args()

    rows = build_candidates(
        max_prompt_chars=args.max_prompt_chars,
        zip_path=Path(args.zip_path),
    )
    output = Path(args.output)
    manifest = Path(args.manifest)
    ensure_artifact_dirs()
    write_jsonl(output, rows)
    write_json(
        manifest,
        {
            "output": str(output),
            "rows": len(rows),
            "candidate_order": "global LongBench QA then Summary source order",
            "datasets": {task: list(datasets) for task, datasets in DATASET_GROUPS.items()},
            "tasks": sorted({row["task"] for row in rows}),
        },
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False))
    return 0


def build_candidates(*, max_prompt_chars: int, zip_path: Path = LONG_BENCH_ZIP) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for task, datasets in DATASET_GROUPS.items():
            task_rows = _build_task_candidates(
                archive,
                task=task,
                datasets=datasets,
                max_prompt_chars=max_prompt_chars,
            )
            rows.extend(task_rows)
    return [_with_candidate_identity(row, candidate_order=index) for index, row in enumerate(rows)]


def _build_task_candidates(
    archive: zipfile.ZipFile,
    *,
    task: str,
    datasets: tuple[str, ...],
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = _read_dataset_rows(archive, dataset=dataset, task=task, max_prompt_chars=max_prompt_chars)
        if rows:
            selected.extend(rows)
    if not selected:
        raise RuntimeError(f"{task} 没有可用 LongBench 子集")
    return selected


def _read_dataset_rows(
    archive: zipfile.ZipFile,
    *,
    dataset: str,
    task: str,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    member = f"data/{dataset}.jsonl"
    if member not in archive.namelist():
        return rows
    with archive.open(member) as handle:
        for source_index, raw in enumerate(handle):
            if not raw.strip():
                continue
            source = json.loads(raw.decode("utf-8"))
            prompt = format_longbench_prompt(source, max_chars=max_prompt_chars)
            reference = normalize_text((source.get("answers") or [None])[0])
            if not prompt or not reference:
                continue
            rows.append(
                {
                    "task": task,
                    "prompt": prompt,
                    "reference": reference,
                    "metadata": {
                        "source": "external_labeling_workspace",
                        "source_dataset": f"longbench_{dataset}",
                        "split": "candidate",
                        "risk_family": "unlabeled",
                        "length_bucket": length_bucket(prompt),
                        "language": normalize_text(source.get("language")),
                        "longbench_dataset": dataset,
                        "source_id": normalize_text(source.get("_id")),
                        "source_index": source_index,
                    },
                }
            )
    return rows


def _with_candidate_identity(row: dict[str, Any], *, candidate_order: int) -> dict[str, Any]:
    metadata = dict(row["metadata"])
    dataset = str(metadata["longbench_dataset"])
    source_index = int(metadata["source_index"])
    request_id = f"longbench_{row['task']}_{dataset}_{source_index:06d}"
    prompt_hash = hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()
    candidate_payload = f"{request_id}\n{row['task']}\n{row['prompt']}\n{row['reference']}"
    metadata.update(
        {
            "prompt_hash": prompt_hash,
            "candidate_order": candidate_order,
            "candidate_hash": hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest(),
        }
    )
    return {**row, "request_id": request_id, "metadata": metadata}


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    return (
        "mkdir -p "
        f"'{output.parent}' '{log.parent}' '{pid.parent}'; "
        "nohup setsid env CUDA_VISIBLE_DEVICES=0,1 "
        "conda run -n tailguardkv-base python -m run_util.build_profile_table "
        f"--config '{config_path}' --output '{output}' --no-dry-run "
        f"> '{log}' 2>&1 < /dev/null & echo $! > '{pid}'"
    )


def launch_next_batch(manifests: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Launch at most one GPU-exclusive batch; later batches require a new invocation."""
    for manifest in manifests:
        pid_path = Path(str(manifest["pid_output"]))
        if pid_path.exists() and _pid_is_running(pid_path):
            raise RuntimeError(f"serial runner is still active for {manifest['batch_id']}")
        csv_path = Path(str(manifest["csv_output"]))
        if csv_path.exists():
            write_prescreen_manifest(manifest, merge_batch_measurements(manifest, csv_path))
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


def _pid_is_running(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and serially launch LongBench direct-prescreen batches.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config")
    parser.add_argument("--launch-next", action="store_true")
    args = parser.parse_args()
    manifests = partition_candidates(read_jsonl(Path(args.candidates)), output_root=Path(args.output_root))
    write_partitions(manifests)
    if args.base_config:
        write_batch_configs(manifests, base_config=Path(args.base_config))
    if args.launch_next:
        if not args.base_config:
            raise SystemExit("--launch-next requires --base-config")
        launched = launch_next_batch(manifests)
        print(json.dumps({"launched": launched["batch_id"] if launched else None}, ensure_ascii=False))
    else:
        print(json.dumps({"batches": len(manifests)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import csv
import hashlib
import json
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
    assert "nohup setsid env CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base" in command
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
model:
  pilot_model: /DATACENTER3/zhenxiang.wang/resource/Qwen2.5-7B-Instruct
  cache_dir: /DATACENTER3/zhenxiang.wang/resource/huggingface

experiment:
  type: baseline_quality

profiles:
  adapters:
    - full
    - kivi
    - h2o
  names:
    - full_gpu
    - kivi_4bit_residual32
    - kivi_4bit_residual64
    - kivi_2bit_residual32
    - kivi_2bit_residual64
    - h2o_heavy10_recent10
    - h2o_heavy15_recent15
    - h2o_heavy20_recent20
  specs:
    full_gpu: {exact: true}
    kivi_4bit_residual32: {exact: false}
    kivi_4bit_residual64: {exact: false}
    kivi_2bit_residual32: {exact: false}
    kivi_2bit_residual64: {exact: false}
    h2o_heavy10_recent10: {exact: false}
    h2o_heavy15_recent15: {exact: false}
    h2o_heavy20_recent20: {exact: false}

policies:
  record_rejected_unsafe: true
  names:
    - full_lru

pilot:
  epsilons: [0.05]
  deltas: [0.05]
  memory_budgets_mib: [4900]

data:
  source: fixture
  quality_mode: baseline
  requests: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/longbench_candidates.jsonl
  calibration_fraction: 0.5
  max_requests: 60

direct_prescreen:
  batch_size: 60
  expected_profiles:
    - full_gpu
    - kivi_4bit_residual32
    - kivi_4bit_residual64
    - kivi_2bit_residual32
    - kivi_2bit_residual64
    - h2o_heavy10_recent10
    - h2o_heavy15_recent15
    - h2o_heavy20_recent20
  serial_gpu_exclusive: true
  launcher:
    conda_env: tailguardkv-base
    cuda_visible_devices: "0,1"
    command: "nohup setsid env CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base python -m run_util.build_profile_table"
  batch_root: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/prescreen/longbench

profile_smoke:
  max_new_tokens: 16
  require_ttft: true
  timeout_s: 180
  repeat: 1
  local_files_only: true
  vllm_enforce_eager: true
  vllm_gpu_memory_utilization: 0.75
  vllm_max_model_len: 1024

outputs:
  smoke_profiles: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/longbench_candidates_profiles.csv
  smoke_profile_checks: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/longbench_candidates_profile_smoke.csv
  smoke_policy: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/tmp/longbench_candidates_policy.csv
  smoke_summary: /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/tmp/longbench_candidates_summary.csv
