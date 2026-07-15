from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from metrics import MetricCollector
from profiles.registry import build_profile_adapters
from core_types import ProfileMeasurement, Request


def default_requests() -> list[Request]:
    """没有正式数据集前，先用固定 smoke 请求验证表结构和执行链路。"""

    return [
        Request(
            request_id="smoke_qa_001",
            task="qa",
            prompt="Question: What is KV cache used for in autoregressive decoding?\nAnswer:",
            reference="KV cache stores past key/value tensors to avoid recomputing attention history.",
        ),
        Request(
            request_id="smoke_sum_001",
            task="summary",
            prompt="Summarize: KV cache compression can reduce memory but may hurt tail quality.",
            reference="KV compression saves memory while risking large quality loss on some requests.",
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_profiles(args: argparse.Namespace) -> int:
    adapters = build_profile_adapters(args.adapters)
    rows = [adapter.smoke(timeout_s=args.timeout).to_row() for adapter in adapters]
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output:
        write_csv(Path(args.output), rows)
    return 0 if all(row["ok"] for row in rows) else 1


def build_profile_table(args: argparse.Namespace) -> int:
    adapters = build_profile_adapters(args.adapters)
    requests = default_requests()
    measurements: list[ProfileMeasurement] = []
    for adapter in adapters:
        for spec in adapter.profiles():
            for request in requests:
                measurements.append(adapter.profile(request, spec.name, dry_run=args.dry_run))

    write_csv(Path(args.output), [measurement.to_row() for measurement in measurements])
    summary = MetricCollector().summarize_profiles(measurements)
    print(json.dumps({"output": args.output, "rows": len(measurements), "summary": summary}, indent=2))
    return 0 if all(measurement.ok for measurement in measurements) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV 统一实验入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-profiles", help="检查 full/KIVI/H2O adapter 连通性。")
    check.add_argument("--adapters", nargs="+", default=["full", "kivi", "h2o"])
    check.add_argument("--timeout", type=int, default=120)
    check.add_argument("--output", default="")
    check.set_defaults(func=check_profiles)

    table = subparsers.add_parser("build-profile-table", help="生成 request x profile 统一表。")
    table.add_argument("--adapters", nargs="+", default=["full", "kivi", "h2o"])
    table.add_argument("--output", default="out/profile_tables/smoke_profiles.csv")
    table.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    table.set_defaults(func=build_profile_table)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
