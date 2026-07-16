#!/usr/bin/env python3
"""Prepare TailGuardKV pilot model and request assets.

The generated request JSONL keeps dataset order. It only maps source fields into
the runner schema and writes skipped-count metadata to a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_RESOURCE_ROOT = Path("/DATACENTER3/zhenxiang.wang/resource")
DEFAULT_OUTPUT_ROOT = DEFAULT_RESOURCE_ROOT / "tailguardkv_pilot"
TINYLLAMA_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_LOCAL = DEFAULT_RESOURCE_ROOT / "TinyLlama-1.1B-Chat-v1.0"
LONGBENCH_REPO = "zai-org/LongBench"
LONGBENCH_CONFIGS = ("qasper", "multifieldqa_en", "hotpotqa")
XSUM_REPO = "EdinburghNLP/xsum"
TARGET_PER_TASK = 300
MIN_PER_TASK = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true", help="download TinyLlama smoke model")
    parser.add_argument("--download-data", action="store_true", help="download and prepare pilot request data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tinyllama-dir", type=Path, default=TINYLLAMA_LOCAL)
    parser.add_argument("--target-per-task", type=int, default=TARGET_PER_TASK)
    parser.add_argument("--min-per-task", type=int, default=MIN_PER_TASK)
    parser.add_argument("--hf-endpoint", default="", help="optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_tinyllama(local_dir: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=TINYLLAMA_REPO,
        local_dir=str(local_dir),
    )
    return {"repo_id": TINYLLAMA_REPO, "local_dir": str(Path(path).resolve())}


def download_hf_file(repo_id: str, filename: str, repo_type: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
    )


def load_longbench_config(config: str, output_root: Path) -> tuple[list[dict[str, Any]], str]:
    zip_path = download_hf_file(
        LONGBENCH_REPO,
        "data.zip",
        "dataset",
        output_root / "hf_downloads" / "LongBench",
    )
    member = f"data/{config}.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        rows = [json.loads(line.decode("utf-8")) for line in archive.open(member) if line.strip()]
    return rows, f"data.zip:{member}"


def load_longbench(output_root: Path) -> tuple[list[dict[str, Any]], str, str]:
    errors: dict[str, str] = {}
    for config in LONGBENCH_CONFIGS:
        try:
            rows, split = load_longbench_config(config, output_root)
            return rows, config, split
        except Exception as exc:
            errors[config] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"无法加载 LongBench QA 配置: {errors}")


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def split_label(position: int, total: int) -> str:
    return "calibration" if position < total / 2 else "eval"


def assign_task_splits(requests: list[dict[str, Any]]) -> None:
    total = len(requests)
    for position, request in enumerate(requests):
        request["split"] = split_label(position, total)


def collect_longbench(output_root: Path, target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, config, split = load_longbench(output_root)
    requests: list[dict[str, Any]] = []
    skipped_empty_prompt = 0
    skipped_empty_reference = 0

    for source_index, row in enumerate(rows):
        prompt = str(row.get("input") or "").strip()
        answers = row.get("answers")
        reference = normalize_answer(answers)
        if not prompt:
            skipped_empty_prompt += 1
            continue
        if not reference:
            skipped_empty_reference += 1
            continue
        position = len(requests)
        requests.append(
            {
                "request_id": f"longbench_{config}_{position:06d}",
                "task": "qa_long_context",
                "prompt": prompt,
                "reference": reference,
                "source": LONGBENCH_REPO,
                "dataset_config": config,
                "source_index": source_index,
                "split": "",
                "source_split": split,
                "answers": answers,
            }
        )
        if len(requests) >= target:
            break

    assign_task_splits(requests)
    manifest = {
        "task": "qa_long_context",
        "source": LONGBENCH_REPO,
        "dataset_config": config,
        "source_split": split,
        "target_count": target,
        "request_count": len(requests),
        "skipped_empty_prompt": skipped_empty_prompt,
        "skipped_empty_reference": skipped_empty_reference,
        "source_index_min": requests[0]["source_index"] if requests else None,
        "source_index_max": requests[-1]["source_index"] if requests else None,
        "order": "original dataset order, no shuffle, no resampling, no dedup",
    }
    return requests, manifest


def collect_xsum_from_hub(output_root: Path, target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    split = "validation"
    parquet_path = download_hf_file(
        XSUM_REPO,
        "data/validation-00000-of-00001.parquet",
        "dataset",
        output_root / "hf_downloads" / "xsum",
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    requests: list[dict[str, Any]] = []
    skipped_empty_prompt = 0
    skipped_empty_reference = 0

    for source_index, row in enumerate(rows):
        document = str(row.get("document") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if not document:
            skipped_empty_prompt += 1
            continue
        if not summary:
            skipped_empty_reference += 1
            continue
        position = len(requests)
        requests.append(
            {
                "request_id": f"xsum_{position:06d}",
                "task": "summary",
                "prompt": f"Summarize:\n{document}\nSummary:",
                "reference": summary,
                "source": XSUM_REPO,
                "dataset_config": "default",
                "source_index": source_index,
                "split": "",
                "source_split": split,
            }
        )
        if len(requests) >= target:
            break

    assign_task_splits(requests)
    manifest = {
        "task": "summary",
        "source": XSUM_REPO,
        "dataset_config": "default",
        "source_split": split,
        "target_count": target,
        "request_count": len(requests),
        "skipped_empty_prompt": skipped_empty_prompt,
        "skipped_empty_reference": skipped_empty_reference,
        "source_index_min": requests[0]["source_index"] if requests else None,
        "source_index_max": requests[-1]["source_index"] if requests else None,
        "order": "original dataset order, no shuffle, no resampling, no dedup",
    }
    return requests, manifest


def prepare_data(output_root: Path, target: int, minimum: int) -> dict[str, Any]:
    request_dir = output_root / "requests"
    request_path = request_dir / "longbench_xsum_pilot.jsonl"
    longbench_requests, longbench_manifest = collect_longbench(output_root, target)
    xsum_requests, xsum_manifest = collect_xsum_from_hub(output_root, target)

    for manifest in (longbench_manifest, xsum_manifest):
        if manifest["request_count"] < minimum:
            raise RuntimeError(
                f"{manifest['task']} 样本数不足 {minimum}: {manifest['request_count']}"
            )

    requests = longbench_requests + xsum_requests
    request_dir.mkdir(parents=True, exist_ok=True)
    with request_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "request_jsonl": str(request_path),
        "total_requests": len(requests),
        "tasks": [longbench_manifest, xsum_manifest],
        "calibration_eval_rule": "per task, first 50% calibration and last 50% eval in original order",
        "data_handling": "no shuffle, no random split, no content rewriting; only runner field mapping",
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if not args.download_models and not args.download_data:
        args.download_models = True
        args.download_data = True

    manifest: dict[str, Any] = {"output_root": str(args.output_root)}
    if args.download_models:
        manifest["tinyllama"] = download_tinyllama(args.tinyllama_dir)
    if args.download_data:
        manifest["data"] = prepare_data(args.output_root, args.target_per_task, args.min_per_task)

    write_json(args.output_root / "prepare_pilot_assets_run.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
