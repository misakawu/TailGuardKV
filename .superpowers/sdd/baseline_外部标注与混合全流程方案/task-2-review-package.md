#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from common import (
    RAGHOT_QA_CANDIDATES,
    RAGHOT_QA_SOURCE,
    ensure_artifact_dirs,
    length_bucket,
    normalize_text,
    write_jsonl,
)


PACKING_POLICY_VERSION = "raghot_support_first_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fully evidenced RAGhot QA candidates for external labeling.")
    parser.add_argument("--input", default=str(RAGHOT_QA_SOURCE))
    parser.add_argument("--output", default=str(RAGHOT_QA_CANDIDATES))
    parser.add_argument("--max-context-chars", type=int, default=3000)
    args = parser.parse_args()

    rows = _read_parquet_rows(Path(args.input))
    candidates = build_candidates(rows, max_context_chars=args.max_context_chars)
    ensure_artifact_dirs()
    write_jsonl(Path(args.output), candidates)
    print(json.dumps({"input": args.input, "output": args.output, "rows": len(candidates)}, ensure_ascii=False))
    return 0


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    max_context_chars: int = 3000,
) -> list[dict[str, Any]]:
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = _build_candidate(row, max_context_chars=max_context_chars)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the RAGhot parquet source") from exc
    return pq.read_table(
        path,
        columns=["id", "question", "answer", "context", "supporting_facts"],
    ).to_pylist()


def _build_candidate(row: dict[str, Any], *, max_context_chars: int) -> dict[str, Any] | None:
    source_record_id = normalize_text(row.get("id"))
    question = normalize_text(row.get("question"))
    answer = normalize_text(row.get("answer"))
    if not source_record_id or not question or not answer:
        return None

    packed = _pack_context(
        row.get("context"),
        row.get("supporting_facts"),
        max_context_chars=max_context_chars,
    )
    if packed is None:
        return None
    context, supporting_fact_ids = packed
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    context_pack_hash = _sha256(context)
    payload_hash = _sha256_json({"task": "qa", "prompt": prompt, "reference": answer})
    return {
        "request_id": f"raghot_qa_{source_record_id}",
        "task": "qa",
        "prompt": prompt,
        "reference": answer,
        "metadata": {
            "source": "external_labeling_workspace",
            "source_dataset": "raghot_qa",
            "split": "candidate",
            "risk_family": "unlabeled",
            "length_bucket": length_bucket(prompt),
            "source_record_id": source_record_id,
            "supporting_fact_ids": supporting_fact_ids,
            "context_pack_hash": context_pack_hash,
            "payload_hash": payload_hash,
            "content_payload_hash": payload_hash,
            "packing_policy_version": PACKING_POLICY_VERSION,
        },
    }


def _pack_context(
    context: Any,
    supporting_facts: Any,
    *,
    max_context_chars: int,
) -> tuple[str, list[str]] | None:
    if not isinstance(context, dict) or not isinstance(supporting_facts, dict):
        return None
    titles = context.get("title")
    documents = context.get("sentences")
    fact_titles = supporting_facts.get("title")
    fact_sentence_ids = supporting_facts.get("sent_id")
    if not all(isinstance(value, list) for value in (titles, documents, fact_titles, fact_sentence_ids)):
        return None
    if len(titles) != len(documents) or len(fact_titles) != len(fact_sentence_ids) or not fact_titles:
        return None

    document_rows: list[tuple[str, list[str]]] = []
    title_indices: dict[str, list[int]] = {}
    for document_index, (raw_title, raw_sentences) in enumerate(zip(titles, documents, strict=True)):
        title = normalize_text(raw_title)
        if not title or not isinstance(raw_sentences, list):
            return None
        sentences = [normalize_text(sentence) for sentence in raw_sentences]
        document_rows.append((title, sentences))
        title_indices.setdefault(title, []).append(document_index)

    supporting_positions: set[tuple[int, int]] = set()
    for raw_title, raw_sentence_id in zip(fact_titles, fact_sentence_ids, strict=True):
        title = normalize_text(raw_title)
        if not title:
            return None
        try:
            sentence_id = int(raw_sentence_id)
        except (TypeError, ValueError):
            return None
        matching_documents = title_indices.get(title, [])
        matching_position = next(
            (
                (document_index, sentence_id)
                for document_index in matching_documents
                if 0 <= sentence_id < len(document_rows[document_index][1])
                and document_rows[document_index][1][sentence_id]
            ),
            None,
        )
        if matching_position is None:
            return None
        supporting_positions.add(matching_position)

    ordered_positions = [
        (document_index, sentence_index)
        for document_index, (_, sentences) in enumerate(document_rows)
        for sentence_index, sentence in enumerate(sentences)
        if sentence
    ]
    supporting = [position for position in ordered_positions if position in supporting_positions]
    nonsupporting = [position for position in ordered_positions if position not in supporting_positions]
    context_parts: list[str] = []
    for position in supporting:
        if not _append_sentence(context_parts, _format_sentence(document_rows, position), max_context_chars):
            return None
    for position in nonsupporting:
        if not _append_sentence(context_parts, _format_sentence(document_rows, position), max_context_chars):
            break

    supporting_fact_ids = [
        f"{document_rows[document_index][0]}:{sentence_index}"
        for document_index, sentence_index in supporting
    ]
    return "\n\n".join(context_parts), supporting_fact_ids


def _append_sentence(parts: list[str], sentence: str, max_context_chars: int) -> bool:
    separator_size = 2 if parts else 0
    if len("\n\n".join(parts)) + separator_size + len(sentence) > max_context_chars:
        return False
    parts.append(sentence)
    return True


def _format_sentence(document_rows: list[tuple[str, list[str]]], position: tuple[int, int]) -> str:
    document_index, sentence_index = position
    title, sentences = document_rows[document_index]
    return f"{title}: {sentences[sentence_index]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: dict[str, str]) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_raghot_qa_candidates import build_candidates


def _record(
    *,
    record_id: str = "record-1",
    question: str = "Which supporting facts are retained?",
    answer: str = "The evidence sentences.",
    context: dict[str, object] | None = None,
    supporting_facts: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "question": question,
        "answer": answer,
        "context": context
        or {
            "title": ["Doc B", "Doc A"],
            "sentences": [
                ["B supporting sentence.", "B distractor sentence."],
                ["A distractor sentence.", "A supporting sentence."],
            ],
        },
        "supporting_facts": supporting_facts
        or {
            "title": ["Doc A", "Doc B"],
            "sent_id": [1, 0],
        },
    }


def test_fully_evidenced_record_has_stable_hashes_and_complete_provenance() -> None:
    record = _record()

    first, second = build_candidates([record]), build_candidates([record])

    assert first == second
    assert len(first) == 1
    candidate = first[0]
    assert candidate["request_id"] == "raghot_qa_record-1"
    assert candidate["task"] == "qa"
    assert candidate["reference"] == "The evidence sentences."
    assert candidate["prompt"] == (
        "Context:\n"
        "Doc B: B supporting sentence.\n\n"
        "Doc A: A supporting sentence.\n\n"
        "Doc B: B distractor sentence.\n\n"
        "Doc A: A distractor sentence.\n\n"
        "Question:\nWhich supporting facts are retained?\nAnswer:"
    )
    metadata = candidate["metadata"]
    assert metadata["source_dataset"] == "raghot_qa"
    assert metadata["source_record_id"] == "record-1"
    assert metadata["supporting_fact_ids"] == ["Doc B:0", "Doc A:1"]
    assert metadata["packing_policy_version"] == "raghot_support_first_v1"
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["context_pack_hash"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["payload_hash"])


def test_rejects_unlocatable_or_over_limit_evidence_and_empty_required_fields() -> None:
    missing_title = _record(supporting_facts={"title": ["Unknown"], "sent_id": [0]})
    missing_sentence = _record(supporting_facts={"title": ["Doc A"], "sent_id": [9]})
    over_limit = _record(
        context={"title": ["Doc A"], "sentences": [["evidence " * 20]]},
        supporting_facts={"title": ["Doc A"], "sent_id": [0]},
    )
    empty_id = _record(record_id=" ")
    empty_question = _record(question=" ")
    empty_answer = _record(answer=" ")

    assert build_candidates([missing_title]) == []
    assert build_candidates([missing_sentence]) == []
    assert build_candidates([over_limit], max_context_chars=10) == []
    assert build_candidates([empty_id, empty_question, empty_answer]) == []


def test_nonsupporting_sentences_are_appended_after_all_supporting_sentences() -> None:
    candidate = build_candidates([_record()])[0]
    context = candidate["prompt"].split("\n\nQuestion:\n", maxsplit=1)[0]

    assert context.index("Doc B: B supporting sentence.") < context.index("Doc A: A supporting sentence.")
    assert context.index("Doc A: A supporting sentence.") < context.index("Doc B: B distractor sentence.")
    assert context.index("Doc B: B distractor sentence.") < context.index("Doc A: A distractor sentence.")
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
