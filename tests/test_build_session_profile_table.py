from __future__ import annotations

import argparse
import csv
from pathlib import Path
from unittest.mock import patch

from run_util.build_profile_table import build_profile_table
from run_util.core_types import ProfileMeasurement, ProfileSpec, Request


def test_build_profile_table_writes_session_keyed_rows_and_trace(tmp_path: Path) -> None:
    profile_csv = tmp_path / "profiles.csv"
    trace_csv = tmp_path / "trace.csv"
    config_path = tmp_path / "config.yaml"
    fixture_path = Path("data/fixtures/sharegpt_sessions.json")
    config_path.write_text(
        "\n".join(
            [
                "pilot:",
                "  epsilons: [0.05]",
                "  deltas: [0.05]",
                "  memory_budgets_mib: [64]",
                "data:",
                "  source: sharegpt",
                f"  requests: {fixture_path}",
                "  calibration_fraction: 0.5",
                "profiles:",
                "  adapters: [full]",
                "  names: [full_gpu]",
                "  specs:",
                "    full_gpu:",
                "      family: full",
                "      exact: true",
                "runtime:",
                "  max_requests: 2",
                "outputs:",
                f"  smoke_profiles: {profile_csv}",
                f"  smoke_profile_trace: {trace_csv}",
            ]
        ),
        encoding="utf-8",
    )

    code = build_profile_table(
        argparse.Namespace(
            config=str(config_path),
            adapters=None,
            output=str(profile_csv),
            import_measurements="",
            dry_run=True,
        )
    )

    rows = list(csv.DictReader(profile_csv.open(encoding="utf-8")))
    trace = list(csv.DictReader(trace_csv.open(encoding="utf-8")))

    assert code == 0
    assert {"session_id", "turn_index", "kv_incremental_mib", "kv_cumulative_mib"} <= set(rows[0])
    session_turns = [row["turn_index"] for row in rows if row["session_id"] == "session_000" and row["profile"] == "full_gpu"]
    assert session_turns[:2] == ["0", "1"]
    assert any(item["event"] in {"resident", "evict", "restore", "recompute"} for item in trace)


def test_build_profile_table_reuses_session_runtime_container_across_chunks(tmp_path: Path) -> None:
    class StubAdapter:
        name = "stub"

        def __init__(self) -> None:
            self.session_runtime_ids: list[int] = []
            self.memory_budgets: list[float | None] = []
            self.chunk_markers: list[int] = []

        def profiles(self):
            return (ProfileSpec("kivi_4bit_residual32", "stub", "env", lossy=True),)

        def profile_many(self, requests, profile_name, dry_run=True, session_runtime=None, memory_budget_mib=None):
            assert session_runtime is not None
            chunk_index = int(session_runtime.get("chunk_index", 0)) + 1
            session_runtime["chunk_index"] = chunk_index
            self.session_runtime_ids.append(id(session_runtime))
            self.memory_budgets.append(memory_budget_mib)
            self.chunk_markers.append(chunk_index)
            return [
                ProfileMeasurement(
                    request_id=request.request_id,
                    profile=profile_name,
                    adapter=self.name,
                    ok=True,
                    measured=True,
                    output_text=request.prompt,
                    latency_ms=1.0,
                    ttft_ms=1.0,
                    peak_memory_mib=1.0,
                    kv_cache_memory_mib=1.0,
                    resident_memory_mib=1.0,
                    session_id=request.session_id,
                    turn_index=request.turn_index,
                    kv_incremental_mib=1.0,
                    kv_cumulative_mib=float(chunk_index),
                    resident_kv_mib_before=float(chunk_index - 1),
                    resident_kv_mib_after=float(chunk_index),
                    restore_ms=0.0,
                    recompute_ms=0.0,
                    evicted_kv_mib=0.0,
                    budget_hit=False,
                    extra={"task": request.task, "length_bucket": "short", "split": request.metadata.get("split", "eval")},
                )
                for request in requests
            ]

    stub = StubAdapter()
    requests = [
        Request(f"r{index}", "qa", f"prompt {index}", session_id="s1", turn_index=index, metadata={"split": "eval"})
        for index in range(12)
    ]
    with (
        patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["kivi_4bit_residual32"]}}),
        patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
        patch("run_util.build_profile_table.config_profiles", return_value=["kivi_4bit_residual32"]),
        patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1, "memory_budget_mib": 64.0}),
        patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
        patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
    ):
        code = build_profile_table(
            argparse.Namespace(
                config="config.yaml",
                adapters=None,
                output=str(tmp_path / "profiles.csv"),
                import_measurements="",
                dry_run=False,
            )
        )

    assert code == 0
    assert stub.chunk_markers == [1, 2]
    assert len(set(stub.session_runtime_ids)) == 1
    assert stub.memory_budgets == [64.0, 64.0]
