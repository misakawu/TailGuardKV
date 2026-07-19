from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from typing import Any


KNOWN_SETUP_ERRORS = (FileNotFoundError, ValueError)


def print_error(error: BaseException, **extra: Any) -> None:
    payload = {"ok": False, "error": str(error), **extra}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_command(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    try:
        return int(func(args))
    except KNOWN_SETUP_ERRORS as exc:
        print_error(exc)
        return 2


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pilot.yaml")


def add_adapters_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapters", nargs="+")


def add_profile_table_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--adapters", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--import-measurements", default="")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--measurements", default="out/profile_tables/smoke_profiles.csv")
    parser.add_argument("--output")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--policy-config")
    parser.add_argument("--epsilon")
    parser.add_argument("--delta")
    parser.add_argument("--memory-budget-mib")
    parser.add_argument("--use-pandas-replay", action="store_true")
    parser.add_argument("--allow-dry-run-replay", action="store_true")


def add_reproduce_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--adapters", nargs="+")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--smoke-output")
    parser.add_argument("--profile-output")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)


def first_number(
    value: Any,
    values: Any,
    *,
    default: float,
    name: str,
) -> float:
    if value is not None:
        return finite_float(value, name)
    if values:
        if isinstance(values, str):
            return finite_float(values, name)
        try:
            return finite_float(next(iter(values)), name)
        except StopIteration:
            return default
        except TypeError:
            return finite_float(values, name)
    return default


def finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是合法数值: {value}") from exc
    if math.isnan(parsed):
        raise ValueError(f"{name} 不能是 NaN")
    return parsed
