from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

from run_util.core_types import PolicyRunRecord
from run_util.experiment_common import write_csv
from run_util.run_policies import _policy_rows_with_provenance
from run_util.session_aggregation import (
    BOOTSTRAP_CI_COLUMNS,
    aggregate_policy_csvs,
    bootstrap_ci_columns,
    parse_sweep_filename,
    summarize_cells,
    write_events_csv,
    write_session_points_csv,
    write_summary_csv,
)
from metrics import session_block_bootstrap_ci


SESSION27_CELL = "session27_policy_eps0p05_delta0p05_mem1000.csv"
PROVENANCE_CONFIG = "configs/pilot_diagnostic_session27.yaml"
PROVENANCE_RUN_DIR = "out/session27_diagnostic"


def _record(
    policy: str,
    request_id: str,
    *,
    session_id: str,
    turn: int,
    ttft_ms: float,
    quality_loss: float = 0.0,
    restore_ms: float = 0.0,
    recompute_ms: float = 0.0,
    queue_delay_ms: float = 0.0,
    evicted_kv_mib: float = 0.0,
    budget_hit: bool = False,
    kv_cache_memory_mib: float = 10.0,
) -> PolicyRunRecord:
    return PolicyRunRecord(
        policy=policy,
        request_id=request_id,
        action_profile="full_gpu",
        ok=True,
        measured=True,
        backend_name="online_qwen",
        session_id=session_id,
        turn_index=turn,
        task="qa",
        length_bucket="short",
        ttft_ms=ttft_ms,
        latency_ms=ttft_ms,
        kv_cache_memory_mib=kv_cache_memory_mib,
        quality_loss=quality_loss,
        restore_ms=restore_ms,
        recompute_ms=recompute_ms,
        queue_delay_ms=queue_delay_ms,
        evicted_kv_mib=evicted_kv_mib,
        budget_hit=budget_hit,
        backend_budget_hit=budget_hit,
        exact=True,
    )


def _write_cell(policy_dir: Path, records: list[PolicyRunRecord], *, run_dir: str = PROVENANCE_RUN_DIR) -> Path:
    path = policy_dir / SESSION27_CELL
    write_csv(
        path,
        _policy_rows_with_provenance(
            records,
            {"data": {"diagnostic_only": True}},
            source_config=PROVENANCE_CONFIG,
            run_dir=run_dir,
        ),
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_parse_session27_sweep_filename() -> None:
    parsed = parse_sweep_filename("session27_policy_eps0p05_delta0p05_mem3699.csv")
    assert parsed == {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 3699.0}


def test_merged_overall_p95_is_p95_over_all_turn_records() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        batch_a = root / "batch_a"
        batch_b = root / "batch_b"
        batch_a.mkdir()
        batch_b.mkdir()
        turns_a = [_record("full_lru", f"sa_t{i}", session_id="sa", turn=i, ttft_ms=10.0) for i in range(100)]
        turns_b = [_record("full_lru", f"sb_t{i}", session_id="sb", turn=i, ttft_ms=2000.0) for i in range(2)]
        path_a = _write_cell(batch_a, turns_a)
        path_b = _write_cell(batch_b, turns_b)

        cells = aggregate_policy_csvs([path_a, path_b])
        assert len(cells) == 1
        assert len(cells[0].records) == 102
        rows = summarize_cells(cells)
        assert len(rows) == 1
        row = rows[0]
        assert float(row["count"]) == 102.0
        assert row["policy"] == "full_lru"

        values = sorted([10.0] * 100 + [2000.0] * 2)
        expected = values[int(round((len(values) - 1) * 0.95))]
        merged_p95 = float(row["p95_ttft_ms"])
        assert merged_p95 == expected

        # 两个 batch 各自 P95 的平均不是合并总体 P95。
        assert (10.0 + 2000.0) / 2.0 != merged_p95


def test_same_cell_across_directories_rejects_mismatched_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        first = root / "run_a"
        second = root / "run_b"
        first.mkdir()
        second.mkdir()
        records = [
            _record("full_lru", "s1_t0", session_id="s1", turn=0, ttft_ms=10.0),
            _record("full_lru", "s1_t1", session_id="s1", turn=1, ttft_ms=12.0),
        ]
        _write_cell(first, records)
        _write_cell(second, records, run_dir="out/other_run")

        import pytest

        with pytest.raises(ValueError, match="run_dir provenance 不一致"):
            aggregate_policy_csvs([first / SESSION27_CELL, second / SESSION27_CELL])


def test_events_csv_counts_match_hand_calculation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dir = Path(tmpdir) / "policy_tables"
        policy_dir.mkdir()
        records = [
            _record(
                "full_lru",
                "s1_t0",
                session_id="s1",
                turn=0,
                ttft_ms=5.0,
                restore_ms=0.0,
                recompute_ms=0.0,
                queue_delay_ms=0.0,
                evicted_kv_mib=0.0,
                budget_hit=False,
            ),
            _record(
                "full_lru",
                "s1_t1",
                session_id="s1",
                turn=1,
                ttft_ms=8.0,
                quality_loss=0.03,
                restore_ms=12.0,
                recompute_ms=3.0,
                queue_delay_ms=5.0,
                evicted_kv_mib=2.0,
                budget_hit=True,
            ),
        ]
        path = _write_cell(policy_dir, records)
        cells = aggregate_policy_csvs([path])
        rows = summarize_cells(cells)
        events_path = write_events_csv(rows, policy_dir / "session27_events.csv")

        [event] = _read_csv(events_path)
        assert float(event["count"]) == 2.0
        assert float(event["ok_count"]) == 2.0
        assert float(event["session_count"]) == 1.0
        assert float(event["budget_hit_count"]) == 1.0
        assert float(event["budget_hit_rate"]) == 0.5
        assert float(event["restore_count"]) == 1.0
        assert float(event["restore_time_ms_total"]) == 12.0
        assert float(event["recompute_count"]) == 1.0
        assert float(event["recompute_time_ms_total"]) == 3.0
        assert float(event["queue_event_count"]) == 1.0
        assert float(event["queue_delay_ms_mean"]) == 2.5
        assert float(event["evict_event_count"]) == 1.0
        assert event["session_reuse_evidence"] == "True"
        assert event["backend_event_evidence"] == "True"


def test_session_points_csv_has_per_session_aggregates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dir = Path(tmpdir) / "policy_tables"
        policy_dir.mkdir()
        records = [
            _record("full_lru", "s1_t0", session_id="s1", turn=0, ttft_ms=10.0),
            _record("full_lru", "s1_t1", session_id="s1", turn=1, ttft_ms=30.0),
            _record("full_lru", "s2_t0", session_id="s2", turn=0, ttft_ms=20.0),
        ]
        path = _write_cell(policy_dir, records)
        rows = summarize_cells(aggregate_policy_csvs([path]))
        points_path = write_session_points_csv(rows, policy_dir / "session27_session_points.csv")

        points = _read_csv(points_path)
        assert len(points) == 2
        by_session = {point["session_id"]: point for point in points}
        assert float(by_session["s1"]["p95_ttft_ms"]) == 30.0
        assert float(by_session["s1"]["mean_ttft_ms"]) == 20.0
        assert float(by_session["s2"]["p95_ttft_ms"]) == 20.0
        assert set(points[0]) == {
            "policy",
            "memory_budget_mib",
            "epsilon",
            "delta",
            "session_id",
            "count",
            "p95_ttft_ms",
            "mean_ttft_ms",
            "p95_quality_loss",
            "mean_quality_loss",
            "mean_kv_cache_memory_mib",
            "violation_rate",
        }


def test_summary_csv_includes_bootstrap_ci_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dir = Path(tmpdir) / "policy_tables"
        policy_dir.mkdir()
        records = []
        for session_id in ("s1", "s2", "s3"):
            for turn in range(2):
                records.append(
                    _record(
                        "full_lru",
                        f"{session_id}_t{turn}",
                        session_id=session_id,
                        turn=turn,
                        ttft_ms=10.0 + float(turn),
                        quality_loss=0.02 + 0.01 * turn,
                    )
                )
        path = _write_cell(policy_dir, records)
        rows = summarize_cells(aggregate_policy_csvs([path]))
        summary_path = write_summary_csv(rows, policy_dir / "session27_total_summary.csv")

        fieldnames = _read_csv(summary_path)[0].keys()
        for column in BOOTSTRAP_CI_COLUMNS:
            assert column in fieldnames
        [row] = _read_csv(summary_path)
        for low_column, high_column in (
            bootstrap_ci_columns("p95_ttft_ms"),
            bootstrap_ci_columns("mean_ttft_ms"),
            bootstrap_ci_columns("mean_quality_loss"),
            bootstrap_ci_columns("p95_quality_loss"),
            bootstrap_ci_columns("violation_rate"),
        ):
            assert row[low_column] != ""
            assert row[high_column] != ""
            assert float(row[low_column]) <= float(row[high_column])


def test_session_block_bootstrap_uses_blocks_not_turns() -> None:
    one_session = [
        _record("full_lru", f"s1_t{i}", session_id="s1", turn=i, ttft_ms=float(10 + i))
        for i in range(8)
    ]
    # 只有 1 个 block（session），即使有多个逐轮值也必须返回 NaN。
    low, high = session_block_bootstrap_ci(one_session, "mean_ttft_ms")
    assert math.isnan(low) and math.isnan(high)


def test_session_block_bootstrap_deterministic_with_fixed_seed() -> None:
    records = []
    for session_index in range(4):
        session_id = f"s{session_index}"
        for turn in range(3):
            records.append(
                _record(
                    "full_lru",
                    f"{session_id}_t{turn}",
                    session_id=session_id,
                    turn=turn,
                    ttft_ms=float(50 + session_index * 10 + turn),
                    quality_loss=0.01 * (session_index + 1),
                )
            )
    first = session_block_bootstrap_ci(records, "p95_ttft_ms", seed=20260906)
    second = session_block_bootstrap_ci(records, "p95_ttft_ms", seed=20260906)
    assert first == second
    assert first[0] <= first[1]
    loss_ci = session_block_bootstrap_ci(records, "violation_rate", epsilon=0.05, seed=20260906)
    assert 0.0 <= loss_ci[0] <= loss_ci[1] <= 1.0
