from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_common import config_adapters, config_runtime, load_config, write_csv
from profiles.registry import build_profile_adapters
from run_cli_common import add_adapters_argument, add_config_argument, run_command


def check_profiles(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = args.output or config.get("outputs", {}).get("smoke_profile_checks", "")
    adapters = build_profile_adapters(args.adapters or config_adapters(config), config_runtime(config))
    rows = [adapter.smoke(timeout_s=args.timeout).to_row() for adapter in adapters]
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    if output:
        write_csv(Path(output), rows)
    return 0 if all(row["ok"] for row in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 full/KIVI/H2O adapter 连通性。")
    add_config_argument(parser)
    add_adapters_argument(parser)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    return run_command(check_profiles, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
