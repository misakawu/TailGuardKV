from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_util.experiment_common import json_ready, load_config


def _float_list(values: list[object]) -> list[float]:
    return [float(value) for value in values]


def load_sweep_grid(config_path: str | Path) -> dict[str, list[float]]:
    config = load_config(Path(config_path))
    pilot = config.get("pilot", {})
    if not isinstance(pilot, dict):
        raise ValueError(f"配置缺少 pilot 段: {config_path}")
    return {
        "memory_budgets_mib": _float_list(list(pilot.get("memory_budgets_mib", []))),
        "epsilons": _float_list(list(pilot.get("epsilons", []))),
        "deltas": _float_list(list(pilot.get("deltas", []))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 baseline wide sweep 的网格配置。")
    parser.add_argument("--config", default="configs/baseline_wide_sweep.yaml")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(json_ready(load_sweep_grid(args.config)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
