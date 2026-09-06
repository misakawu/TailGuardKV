from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from run_util.core_types import ProfileMeasurement
from run_util.io_utils import read_measurements


SESSION27_SPLIT_SEED = 20260906
_BUDGET_QUANTILES = (("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90))


def derive_full_no_eviction_budgets(
    measurements: list[ProfileMeasurement],
    *,
    full_profile: str = "full_gpu",
    split_seed: int = SESSION27_SPLIT_SEED,
) -> dict[str, Any]:
    """Derive session27 B values from full-profile global KV occupancy without eviction."""
    selected = [row for row in measurements if row.profile == full_profile and row.ok and row.measured]
    if not selected:
        raise ValueError(f"没有可用的 full profile 测量记录: {full_profile}")

    ordered = sorted(
        selected,
        key=lambda row: (_arrival_index(row), row.session_id or row.request_id, row.turn_index, row.request_id),
    )
    resident_by_session: dict[str, float] = {}
    occupancy_sequence: list[dict[str, Any]] = []
    for row in ordered:
        session_id = row.session_id or row.request_id
        cumulative_mib = _cumulative_mib(row)
        resident_by_session[session_id] = cumulative_mib
        occupancy_sequence.append(
            {
                "request_id": row.request_id,
                "session_id": session_id,
                "turn_index": row.turn_index,
                "arrival_index": _arrival_index(row),
                "occupancy_mib": sum(resident_by_session.values()),
                "diagnostic_only": True,
            }
        )

    occupancies = [float(item["occupancy_mib"]) for item in occupancy_sequence]
    percentiles = {name: _linear_percentile(occupancies, quantile) for name, quantile in _BUDGET_QUANTILES}
    return {
        "diagnostic_only": True,
        "quality_status": "risk_evidence_insufficient",
        "violation_status": "risk_evidence_insufficient",
        "split_seed": split_seed,
        "full_profile": full_profile,
        "eviction_policy": "none",
        "occupancy_sequence": occupancy_sequence,
        "percentiles_mib": percentiles,
        "memory_budgets_mib": [percentiles[name] for name, _ in _BUDGET_QUANTILES],
    }


def _arrival_index(row: ProfileMeasurement) -> int:
    raw = row.extra.get("arrival_index")
    return row.turn_index if raw in {None, ""} else int(float(raw))


def _cumulative_mib(row: ProfileMeasurement) -> float:
    value = row.kv_cumulative_mib
    if value is None:
        value = row.resident_kv_mib_after
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"full profile 缺少有效 cumulative KV: {row.request_id}")
    return float(value)


def _linear_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("occupancy 序列为空")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 full 无驱逐 KV occupancy 生成 session27 B 档位")
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-profile", default="full_gpu")
    parser.add_argument("--split-seed", type=int, default=SESSION27_SPLIT_SEED)
    args = parser.parse_args()

    payload = derive_full_no_eviction_budgets(
        read_measurements(Path(args.measurements), require_quality_loss=False),
        full_profile=args.full_profile,
        split_seed=args.split_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
