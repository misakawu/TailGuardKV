from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/DATACENTER3/zhenxiang.wang/work/TailGuardKV")
LONG_BENCH_ZIP = Path("/DATACENTER3/zhenxiang.wang/resource/LongBench/data.zip")
SHAREGPT_SOURCE = Path("/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/requests/sharegpt_longbench_multiturn_pilot.jsonl")
RAGHOT_QA_SOURCE = Path("/DATACENTER3/zhenxiang.wang/resource/RAGhot_QA/validation-00000-of-00001.parquet")

ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts"
CANDIDATE_ROOT = ARTIFACT_ROOT / "candidates"
MEASUREMENT_ROOT = ARTIFACT_ROOT / "measurements"
FIXTURE_ROOT = ARTIFACT_ROOT / "fixtures"
MANIFEST_ROOT = ARTIFACT_ROOT / "manifests"

LONG_BENCH_CANDIDATES = CANDIDATE_ROOT / "longbench_candidates.jsonl"
SHAREGPT_CANDIDATES = CANDIDATE_ROOT / "sharegpt_session_candidates.jsonl"
RAGHOT_QA_CANDIDATES = CANDIDATE_ROOT / "raghot_qa_candidates.jsonl"
LONG_BENCH_MEASUREMENTS = MEASUREMENT_ROOT / "longbench_candidates_profiles.csv"
SHAREGPT_MEASUREMENTS = MEASUREMENT_ROOT / "sharegpt_session_candidates_profiles.csv"
BASELINE_QUALITY_FIXTURE = FIXTURE_ROOT / "baseline_quality_external.jsonl"
BASELINE_SESSION_FIXTURE = FIXTURE_ROOT / "baseline_session_external.jsonl"
BASELINE_QUALITY_MANIFEST = MANIFEST_ROOT / "baseline_quality_external_manifest.json"
BASELINE_SESSION_MANIFEST = MANIFEST_ROOT / "baseline_session_external_manifest.json"

MAINSTREAM_KIVI_PROFILES = (
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
)
MAINSTREAM_H2O_PROFILES = (
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
SENSITIVE_THRESHOLD = 0.05
LOW_RISK_THRESHOLD = 0.01
TIE_EPSILON = 0.01


def ensure_repo_import_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def ensure_artifact_dirs() -> None:
    for path in (CANDIDATE_ROOT, MEASUREMENT_ROOT, FIXTURE_ROOT, MANIFEST_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def length_bucket(text: str) -> str:
    size = len(text)
    if size < 512:
        return "short"
    if size < 2048:
        return "medium"
    if size < 8192:
        return "long"
    return "xl"
