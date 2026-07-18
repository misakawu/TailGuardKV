from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_common import config_adapters, json_ready, load_config
from run_build_profile_table import build_profile_table
from run_check_profiles import check_profiles
from run_cli_common import add_reproduce_arguments, KNOWN_SETUP_ERRORS, run_command


def reproduce_profiles(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    adapters = args.adapters or config_adapters(config)
    outputs = config.get("outputs", {})
    smoke_args = argparse.Namespace(
        config=args.config,
        adapters=adapters,
        timeout=args.timeout,
        output=args.smoke_output or outputs.get("smoke_profile_checks", "out/profile_tables/profile_smoke.csv"),
    )
    table_args = argparse.Namespace(
        config=args.config,
        adapters=adapters,
        output=args.profile_output or outputs.get("smoke_profiles", "out/profile_tables/smoke_profiles.csv"),
        dry_run=args.dry_run,
        import_measurements="",
    )
    steps = []
    try:
        smoke_code = check_profiles(smoke_args)
    except KNOWN_SETUP_ERRORS as exc:
        smoke_code = 2
        steps.append({"step": "check-profiles", "ok": False, "return_code": smoke_code, "error": str(exc)})
    else:
        steps.append({"step": "check-profiles", "ok": smoke_code == 0, "return_code": smoke_code, "output": smoke_args.output})

    try:
        table_code = build_profile_table(table_args)
    except KNOWN_SETUP_ERRORS as exc:
        table_code = 2
        steps.append({"step": "build-profile-table", "ok": False, "return_code": table_code, "error": str(exc)})
    else:
        steps.append({"step": "build-profile-table", "ok": table_code == 0, "return_code": table_code, "output": table_args.output})

    return_code = 0 if smoke_code == 0 and table_code == 0 else (2 if 2 in {smoke_code, table_code} else 1)
    print(
        json.dumps(
            json_ready({"ok": return_code == 0, "return_code": return_code, "steps": steps}),
            ensure_ascii=False,
            indent=2,
        )
    )
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复现 full/KIVI/H2O adapter smoke 和统一 profile 表。")
    add_reproduce_arguments(parser)
    return parser


def main() -> int:
    return run_command(reproduce_profiles, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
