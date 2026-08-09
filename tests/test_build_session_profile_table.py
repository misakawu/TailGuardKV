from __future__ import annotations

import argparse
import csv
from pathlib import Path
from unittest.mock import patch

from profiles.base import PersistentWorkerFatalError
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
        patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1, "memory_budget_mib": 64.0, "profile_chunk_size": 4}),
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
    assert stub.chunk_markers == [1, 2, 3]
    assert len(set(stub.session_runtime_ids)) == 1
    assert stub.memory_budgets == [64.0, 64.0, 64.0]


def test_build_profile_table_uses_persistent_worker_when_enabled(tmp_path: Path) -> None:
    class StubWorker:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class StubAdapter:
        name = "full"
        env = "tailguardkv-base"

        def __init__(self) -> None:
            self.worker_ids: list[int] = []
            self.chunk_markers: list[int] = []

        def profiles(self):
            return (ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True),)

        def profile_many(
            self,
            requests,
            profile_name,
            dry_run=True,
            session_runtime=None,
            memory_budget_mib=None,
            persistent_worker=None,
        ):
            assert dry_run is False
            assert persistent_worker is not None
            assert session_runtime is not None
            self.worker_ids.append(id(persistent_worker))
            chunk_index = int(session_runtime.get("chunk_index", 0)) + 1
            session_runtime["chunk_index"] = chunk_index
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

    stub_worker = StubWorker()
    stub = StubAdapter()
    requests = [
        Request(f"r{index}", "qa", f"prompt {index}", session_id="s1", turn_index=index, metadata={"split": "eval"})
        for index in range(6)
    ]
    with (
        patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full"], "names": ["full_gpu"]}}),
        patch("run_util.build_profile_table.config_adapters", return_value=["full"]),
        patch(
            "run_util.build_profile_table.config_runtime",
            return_value={"repeat": 1, "memory_budget_mib": 64.0, "profile_chunk_size": 2, "use_persistent_workers": True},
        ),
        patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
        patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
        patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
        patch("run_util.build_profile_table.create_persistent_worker", return_value=stub_worker) as create_worker,
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
    assert create_worker.call_count == 1
    assert len(set(stub.worker_ids)) == 1
    assert stub.chunk_markers == [1, 2, 3]
    assert stub_worker.closed is True


def test_build_profile_table_skips_persistent_worker_when_disabled(tmp_path: Path) -> None:
    class StubAdapter:
        name = "full"

        def __init__(self) -> None:
            self.persistent_workers: list[object | None] = []

        def profiles(self):
            return (ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True),)

        def profile_many(
            self,
            requests,
            profile_name,
            dry_run=True,
            session_runtime=None,
            memory_budget_mib=None,
            persistent_worker=None,
        ):
            self.persistent_workers.append(persistent_worker)
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
                    kv_cumulative_mib=1.0,
                    resident_kv_mib_before=0.0,
                    resident_kv_mib_after=1.0,
                    restore_ms=0.0,
                    recompute_ms=0.0,
                    evicted_kv_mib=0.0,
                    budget_hit=False,
                    extra={"task": request.task, "length_bucket": "short", "split": request.metadata.get("split", "eval")},
                )
                for request in requests
            ]

    stub = StubAdapter()
    requests = [Request("r0", "qa", "prompt 0", session_id="s1", turn_index=0, metadata={"split": "eval"})]
    with (
        patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full"], "names": ["full_gpu"]}}),
        patch("run_util.build_profile_table.config_adapters", return_value=["full"]),
        patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
        patch(
            "run_util.build_profile_table.config_runtime",
            return_value={"repeat": 1, "memory_budget_mib": 64.0, "profile_chunk_size": 2, "use_persistent_workers": False},
        ),
        patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
        patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
        patch("run_util.build_profile_table.create_persistent_worker") as create_worker,
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
    create_worker.assert_not_called()
    assert stub.persistent_workers == [None]


def test_build_profile_table_stops_after_persistent_worker_fatal_error(tmp_path: Path) -> None:
    class StubWorker:
        def close(self) -> None:
            return None

    class StubAdapter:
        name = "full"

        def __init__(self) -> None:
            self.calls = 0

        def profiles(self):
            return (ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True),)

        def profile_many(
            self,
            requests,
            profile_name,
            dry_run=True,
            session_runtime=None,
            memory_budget_mib=None,
            persistent_worker=None,
        ):
            del dry_run, session_runtime, memory_budget_mib, persistent_worker
            self.calls += 1
            failed = [
                ProfileMeasurement(
                    request_id=request.request_id,
                    profile=profile_name,
                    adapter=self.name,
                    ok=False,
                    measured=False,
                    session_id=request.session_id,
                    turn_index=request.turn_index,
                    error="CUDA out of memory",
                    extra={"task": request.task, "length_bucket": "short", "split": request.metadata.get("split", "eval")},
                )
                for request in requests
            ]
            raise PersistentWorkerFatalError("CUDA out of memory", failed)

    stub = StubAdapter()
    requests = [
        Request("r0", "qa", "prompt 0", session_id="s1", turn_index=0, metadata={"split": "eval"}),
        Request("r1", "qa", "prompt 1", session_id="s1", turn_index=1, metadata={"split": "eval"}),
    ]
    output_path = tmp_path / "profiles.csv"
    diagnostic_path = tmp_path / "profiles_failed_chunks.csv"
    with (
        patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full"], "names": ["full_gpu"]}}),
        patch("run_util.build_profile_table.config_adapters", return_value=["full"]),
        patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
        patch(
            "run_util.build_profile_table.config_runtime",
            return_value={"repeat": 1, "memory_budget_mib": 64.0, "profile_chunk_size": 2, "use_persistent_workers": True},
        ),
        patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
        patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
        patch("run_util.build_profile_table.create_persistent_worker", return_value=StubWorker()),
    ):
        code = build_profile_table(
            argparse.Namespace(
                config="config.yaml",
                adapters=None,
                output=str(output_path),
                import_measurements="",
                dry_run=False,
            )
        )

    assert code == 2
    assert stub.calls == 1
    assert not output_path.exists()
    assert diagnostic_path.exists()


def test_build_profile_table_closes_previous_persistent_worker_before_next_adapter(tmp_path: Path) -> None:
    close_events: list[str] = []

    class StubWorker:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            self.closed = True
            close_events.append(self.name)

    class FullAdapter:
        name = "full"
        env = "tailguardkv-base"

        def profiles(self):
            return (ProfileSpec("full_gpu", "full", self.env, lossy=False, exact=True),)

        def profile_many(
            self,
            requests,
            profile_name,
            dry_run=True,
            session_runtime=None,
            memory_budget_mib=None,
            persistent_worker=None,
        ):
            assert persistent_worker is not None
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
                    kv_cumulative_mib=1.0,
                    resident_kv_mib_before=0.0,
                    resident_kv_mib_after=1.0,
                    restore_ms=0.0,
                    recompute_ms=0.0,
                    evicted_kv_mib=0.0,
                    budget_hit=False,
                    extra={"task": request.task, "length_bucket": "short", "split": request.metadata.get("split", "eval")},
                )
                for request in requests
            ]

    class KiviAdapter:
        name = "kivi"
        env = "edgekv-kivi"

        def __init__(self, full_worker: StubWorker) -> None:
            self.full_worker = full_worker

        def profiles(self):
            return (ProfileSpec("kivi_4bit_residual32", "kivi", self.env, lossy=True),)

        def profile_many(
            self,
            requests,
            profile_name,
            dry_run=True,
            session_runtime=None,
            memory_budget_mib=None,
            persistent_worker=None,
        ):
            assert persistent_worker is not None
            assert self.full_worker.closed is True
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
                    kv_cumulative_mib=1.0,
                    resident_kv_mib_before=0.0,
                    resident_kv_mib_after=1.0,
                    restore_ms=0.0,
                    recompute_ms=0.0,
                    evicted_kv_mib=0.0,
                    budget_hit=False,
                    extra={"task": request.task, "length_bucket": "short", "split": request.metadata.get("split", "eval")},
                )
                for request in requests
            ]

    full_worker = StubWorker("full")
    kivi_worker = StubWorker("kivi")
    adapters = [FullAdapter(), KiviAdapter(full_worker)]
    requests = [Request("r0", "qa", "prompt 0", session_id="s1", turn_index=0, metadata={"split": "eval"})]

    def create_worker(adapter, runtime):
        del runtime
        return full_worker if adapter.name == "full" else kivi_worker

    with (
        patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full", "kivi"], "names": ["full_gpu", "kivi_4bit_residual32"]}}),
        patch("run_util.build_profile_table.config_adapters", return_value=["full", "kivi"]),
        patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu", "kivi_4bit_residual32"]),
        patch(
            "run_util.build_profile_table.config_runtime",
            return_value={"repeat": 1, "memory_budget_mib": 64.0, "profile_chunk_size": 1, "use_persistent_workers": True},
        ),
        patch("run_util.build_profile_table.build_profile_adapters", return_value=adapters),
        patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
        patch("run_util.build_profile_table.create_persistent_worker", side_effect=create_worker),
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
    assert close_events == ["full", "kivi"]
