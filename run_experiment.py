from __future__ import annotations

import argparse

from experiment_common import read_measurements, write_csv
from run_build_profile_table import build_profile_table
from run_check_profiles import check_profiles
from run_cli_common import add_policy_arguments, add_profile_table_arguments, add_reproduce_arguments, run_command
from run_reproduce_profiles import reproduce_profiles
from run_run_policies import run_policies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV 兼容实验入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-profiles", help="检查 full/KIVI/H2O adapter 连通性。")
    check.add_argument("--config", default="configs/pilot.yaml")
    check.add_argument("--adapters", nargs="+")
    check.add_argument("--timeout", type=int, default=120)
    check.add_argument("--output", default="")
    check.set_defaults(func=check_profiles)

    table = subparsers.add_parser("build-profile-table", help="生成 request x profile 统一表。")
    add_profile_table_arguments(table)
    table.set_defaults(func=build_profile_table)

    replay = subparsers.add_parser("run-policies", help="用同一 measured-replay backend 运行策略。")
    add_policy_arguments(replay)
    replay.set_defaults(func=run_policies)

    reproduce = subparsers.add_parser("reproduce-profiles", help="复现 full/KIVI/H2O adapter smoke 和统一 profile 表。")
    add_reproduce_arguments(reproduce)
    reproduce.set_defaults(func=reproduce_profiles)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_command(args.func, args)


if __name__ == "__main__":
    raise SystemExit(main())
