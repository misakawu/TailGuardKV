#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_asset_prepare.prepare_pilot_assets import format_longbench_prompt


OUTPUT_PATH = REPO_ROOT / "data" / "fixtures" / "pilot_qa_summary_requests.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "fixtures" / "pilot_qa_summary_manifest.json"
LONGBENCH_ZIP = Path("/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/hf_downloads/LongBench/data.zip")
RAGHOT_PATH = Path("/DATACENTER3/zhenxiang.wang/resource/RAGhot_QA/validation-00000-of-00001.parquet")
XSUM_PATH = Path("/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/hf_downloads/xsum/data/validation-00000-of-00001.parquet")


def main() -> int:
    qa_short = _collect_raghot_short(limit=100)
    qa_nonshort = _collect_longbench_nonshort(limit=100)
    summary_short, summary_nonshort = _collect_xsum_rows()

    calibration_rows = _assign_split_rows("calibration", qa_short[:50], qa_nonshort[:50])
    calibration_rows += _assign_split_rows("calibration", summary_short[:16], summary_nonshort[:134])
    eval_rows = _assign_split_rows("eval", qa_short[50:100], qa_nonshort[50:100])
    eval_rows += _assign_split_rows("eval", summary_short[16:32], summary_nonshort[134:268])

    rows = calibration_rows + eval_rows
    for arrival_index, row in enumerate(rows):
        row["metadata"]["arrival_index"] = arrival_index

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "request_path": str(OUTPUT_PATH),
                "total_requests": len(rows),
                "tasks": {
                    "qa": {
                        "short": 100,
                        "nonshort": 100,
                        "sources": ["RAGhot_QA validation", "LongBench qasper"],
                    },
                    "summary": {
                        "short": 32,
                        "nonshort": 268,
                        "sources": ["XSum validation"],
                    },
                },
                "splits": {
                    "calibration": 250,
                    "eval": 250,
                },
                "note": "Baseline pilot main data only covers independent QA + summary requests. Session-heavy traces stay in separate ShareGPT configs.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _assign_split_rows(split: str, short_rows: list[dict[str, Any]], nonshort_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _interleave(short_rows, nonshort_rows)
    for row in rows:
        row["metadata"]["split"] = split
    return rows


def _interleave(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_len = max(len(primary), len(secondary))
    for index in range(max_len):
        if index < len(primary):
            rows.append(primary[index])
        if index < len(secondary):
            rows.append(secondary[index])
    return rows


def _collect_longbench_nonshort(*, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(LONGBENCH_ZIP) as archive:
        for raw in archive.open("data/qasper.jsonl"):
            if not raw.strip():
                continue
            source = json.loads(raw.decode("utf-8"))
            prompt = format_longbench_prompt(source, max_chars=6000)
            reference = _normalize_text((source.get("answers") or [None])[0])
            if not prompt or not reference or _length_bucket(prompt) != "nonshort":
                continue
            rows.append(
                {
                    "request_id": f"qa_longbench_{len(rows):06d}",
                    "task": "qa",
                    "prompt": prompt,
                    "reference": reference,
                    "metadata": {
                        "source_dataset": "longbench_qasper",
                        "split": "",
                        "length_bucket": "nonshort",
                    },
                }
            )
            if len(rows) >= limit:
                break
    if len(rows) != limit:
        raise RuntimeError(f"LongBench qasper nonshort QA 数量不足: {len(rows)}")
    return rows


def _collect_raghot_short(*, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in pq.read_table(RAGHOT_PATH).to_pylist():
        question = _normalize_text(source.get("question"))
        reference = _normalize_text(source.get("answer"))
        rag_chunks = _raghot_chunks(source, max_chunks=2)
        context = "\n\n".join(chunk["content"] for chunk in rag_chunks)
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:" if context else f"Question:\n{question}\nAnswer:"
        if not question or not reference or _length_bucket(prompt) != "short":
            continue
        rows.append(
            {
                "request_id": f"qa_raghot_{len(rows):06d}",
                "task": "qa",
                "prompt": prompt,
                "reference": reference,
                "metadata": {
                    "source_dataset": "raghot_qa",
                    "split": "",
                    "length_bucket": "short",
                    "rag_chunks": rag_chunks,
                },
            }
        )
        if len(rows) >= limit:
            break
    if len(rows) != limit:
        raise RuntimeError(f"RAGhot 短 QA 数量不足: {len(rows)}")
    return rows


def _raghot_chunks(source: dict[str, Any], *, max_chunks: int) -> list[dict[str, str]]:
    context = source.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    chunks: list[dict[str, str]] = []
    for index, (title, sentence_list) in enumerate(zip(titles, sentences)):
        merged = " ".join(_normalize_text(sentence) for sentence in sentence_list if _normalize_text(sentence))
        if not merged:
            continue
        content = f"{_normalize_text(title)}: {merged}".strip(": ")
        chunks.append({"chunk_id": f"{source.get('id', 'raghot')}-{index}", "content": content})
        if len(chunks) >= max_chunks:
            break
    return chunks


def _collect_xsum_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    short_rows: list[dict[str, Any]] = []
    nonshort_rows: list[dict[str, Any]] = []
    for source in pq.read_table(XSUM_PATH).to_pylist():
        document = _normalize_text(source.get("document"))
        reference = _normalize_text(source.get("summary"))
        prompt = f"Summarize:\n{document}\nSummary:"
        if not document or not reference:
            continue
        row = {
            "request_id": f"summary_xsum_{len(short_rows) + len(nonshort_rows):06d}",
            "task": "summary",
            "prompt": prompt,
            "reference": reference,
            "metadata": {
                "source_dataset": "xsum",
                "split": "",
                "length_bucket": _length_bucket(prompt),
            },
        }
        if row["metadata"]["length_bucket"] == "short":
            short_rows.append(row)
        else:
            nonshort_rows.append(row)
        if len(short_rows) >= 32 and len(nonshort_rows) >= 268:
            break
    if len(short_rows) < 32 or len(nonshort_rows) < 268:
        raise RuntimeError(f"XSum 长度分组不足: short={len(short_rows)} nonshort={len(nonshort_rows)}")
    return short_rows[:32], nonshort_rows[:268]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _length_bucket(prompt: str) -> str:
    return "short" if len(prompt) < 512 else "nonshort"


if __name__ == "__main__":
    raise SystemExit(main())
