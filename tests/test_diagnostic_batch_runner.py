from __future__ import annotations

import json
import csv
import sys

import pytest
import yaml
from pathlib import Path
from types import SimpleNamespace

import scripts.run_diagnostic_session_batches as diagnostic_runner
import scripts.run_session27_online_sweeps as online_sweeps
from scripts.run_diagnostic_session_batches import (
    SessionBatch,
    is_diagnostic_gate_failure,
    materialize_session_batches,
    run_batches,
    validate_batch_output,
)


def _batch_output(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[SessionBatch, Path]:
    fixture = tmp_path / "fixture.jsonl"
    request_ids = list(dict.fromkeys(str(row["request_id"]) for row in rows))
    fixture.write_text("".join(json.dumps({"request_id": request_id}) + "\n" for request_id in request_ids), encoding="utf-8")
    run_dir = tmp_path / "run"
    profile_dir = run_dir / "profile_tables"
    profile_dir.mkdir(parents=True)
    with (profile_dir / "diagnostic_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return SessionBatch("batch000", fixture, 1, len(rows) // 2), run_dir


def test_materialize_session_batches_keeps_complete_sessions_and_arrival_order(tmp_path: Path) -> None:
    rows = [
        {"request_id": f"s{session}t{turn}", "session_id": f"s{session}", "turn_index": turn,
         "arrival_index": arrival, "task": "chat", "prompt": "p", "reference": "r", "metadata": {"diagnostic_only": True}}
        for arrival, (turn, session) in enumerate((turn, session) for turn in range(5) for session in range(4))
    ]
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    batches = materialize_session_batches(fixture, tmp_path / "batches", sessions_per_batch=3)

    assert [batch.session_count for batch in batches] == [4]
    first = [json.loads(line) for line in batches[0].fixture_path.read_text(encoding="utf-8").splitlines()]
    assert len(first) == 20
    assert {row["session_id"] for row in first} == {"s0", "s1", "s2", "s3"}
    assert [row["arrival_index"] for row in first] == list(range(20))


def test_session27_runners_default_to_two_sessions_per_batch(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    (tmp_path / "run").mkdir()
    observed: dict[str, int] = {}

    def capture_batch_size(
        fixture_path: Path,
        output_root: Path,
        *,
        sessions_per_batch: int,
    ) -> list[SessionBatch]:
        assert fixture_path == fixture
        assert output_root == tmp_path / "run" / "fixtures"
        observed["sessions_per_batch"] = sessions_per_batch
        return []

    monkeypatch.setattr(diagnostic_runner, "materialize_session_batches", capture_batch_size)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_diagnostic_session_batches.py",
            "--fixture",
            str(fixture),
            "--config",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--prepare-only",
        ],
    )

    assert diagnostic_runner.main() == 0
    assert observed["sessions_per_batch"] == 2
    assert online_sweeps.build_parser().parse_args([]).sessions_per_batch == 2


def test_validate_batch_output_accepts_complete_measured_profile_coverage(tmp_path: Path) -> None:
    rows = [
        {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    batch, run_dir = _batch_output(tmp_path, rows)

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is True
    assert status["profile_rows"] == 4


def test_validate_batch_output_normalizes_pressure_trace_request_ids(tmp_path: Path) -> None:
    rows = [
        {
            "request_id": f"{request_id}__pressure_r1_c1",
            "profile": profile,
            "ok": "True",
            "measured": "True",
        }
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    batch, run_dir = _batch_output(tmp_path, rows)
    batch.fixture_path.write_text(
        '{"request_id": "r1"}\n{"request_id": "r2"}\n',
        encoding="utf-8",
    )

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is True


def test_validate_batch_output_rejects_duplicate_profile_coverage(tmp_path: Path) -> None:
    rows = [
        {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    rows.append(rows[0].copy())
    batch, run_dir = _batch_output(tmp_path, rows)

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is False
    assert "duplicate" in status["errors"][0]


def test_is_diagnostic_gate_failure_detects_failed_trace_gate(tmp_path: Path) -> None:
    gate_path = tmp_path / "profile_tables" / "pilot_session_trace_semantics_gate.json"
    gate_path.parent.mkdir()
    gate_path.write_text(json.dumps({"passed": False}), encoding="utf-8")
    summary_path = tmp_path / "policy_tables" / "diagnostic_session27_summary.csv"
    summary_path.parent.mkdir()
    summary_path.write_text(
        "section,name,ok,error\nexperiment,pilot-smoke-measured,False,session trace semantics gate failed\n",
        encoding="utf-8",
    )

    assert is_diagnostic_gate_failure(tmp_path) is True


def test_validate_batch_output_rejects_duplicate_fixture_request_ids(tmp_path: Path) -> None:
    rows = [
        {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    batch, run_dir = _batch_output(tmp_path, rows)
    batch.fixture_path.write_text('{"request_id": "r1"}\n{"request_id": "r1"}\n', encoding="utf-8")

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is False
    assert any("duplicate fixture request_id" in error for error in status["errors"])


def test_validate_batch_output_rejects_empty_fixture_request_ids(tmp_path: Path) -> None:
    rows = [
        {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    batch, run_dir = _batch_output(tmp_path, rows)
    batch.fixture_path.write_text('{"request_id": ""}\n', encoding="utf-8")

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is False
    assert any("empty fixture request_id" in error for error in status["errors"])


def test_validate_batch_output_rejects_fixture_row_count_mismatch(tmp_path: Path) -> None:
    rows = [
        {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    batch, run_dir = _batch_output(tmp_path, rows)
    mismatched_batch = SessionBatch(batch.batch_id, batch.fixture_path, batch.session_count, 3)

    status = validate_batch_output(mismatched_batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is False
    assert any("fixture request count mismatch" in error for error in status["errors"])


def _supervisor_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    root = tmp_path / "supervisor"
    batches: list[dict[str, object]] = []
    for index in range(2):
        batch_id = f"batch{index:03d}"
        fixture = root / "fixtures" / f"{batch_id}.jsonl"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("".join(json.dumps({"request_id": request_id}) + "\n" for request_id in (f"r{index}a", f"r{index}b")), encoding="utf-8")
        run_dir = root / "batch_outputs" / batch_id
        profile_dir = run_dir / "profile_tables"
        profile_dir.mkdir(parents=True)
        with (profile_dir / "diagnostic_session27_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["request_id", "profile", "ok", "measured"])
            writer.writeheader()
            writer.writerows(
                {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
                for request_id in (f"r{index}a", f"r{index}b")
                for profile in ("full_gpu", "lossy")
            )
        trace_dir = run_dir / "session_traces"
        trace_dir.mkdir()
        (trace_dir / "diagnostic_session27_trace.csv").write_text("request_id\n" + f"r{index}a\n", encoding="utf-8")
        batches.append(
            {
                "batch_id": batch_id,
                "fixture": str(fixture),
                "config": str(root / "configs" / f"{batch_id}.yaml"),
                "run_dir": str(run_dir),
                "sessions": 1,
                "requests": 2,
            }
        )
    return {"diagnostic_only": True, "profile_names": ["full_gpu", "lossy"], "batches": batches}, root


def _write_supervisor_output(
    item: dict[str, object],
    *,
    failed_gate: str | None = None,
    summary_error: str = "",
    profile_fieldnames: list[str] | None = None,
    malformed_profile_row: bool = False,
) -> None:
    run_dir = Path(item["run_dir"])
    request_ids = [json.loads(line)["request_id"] for line in Path(item["fixture"]).read_text(encoding="utf-8").splitlines()]
    profile_dir = run_dir / "profile_tables"
    profile_dir.mkdir(parents=True, exist_ok=True)
    with (profile_dir / "diagnostic_session27_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=profile_fieldnames or ["request_id", "profile", "ok", "measured"],
        )
        writer.writeheader()
        writer.writerows(
            {"request_id": request_id, "profile": profile, "ok": "True", "measured": "True"}
            for request_id in request_ids
            for profile in ("full_gpu", "lossy")
        )
    trace_dir = run_dir / "session_traces"
    trace_dir.mkdir(exist_ok=True)
    (trace_dir / "diagnostic_session27_trace.csv").write_text("request_id\n" + f"{request_ids[0]}\n", encoding="utf-8")
    if malformed_profile_row:
        profile_path = profile_dir / "diagnostic_session27_profiles.csv"
        lines = profile_path.read_text(encoding="utf-8").splitlines()
        profile_path.write_text("\n".join([lines[0], f"{lines[1]},unexpected", *lines[2:]]) + "\n", encoding="utf-8")
    if failed_gate:
        gate_name = "pilot_session_risk_signal_gate.json" if failed_gate == "risk" else "pilot_session_trace_semantics_gate.json"
        (profile_dir / gate_name).write_text(json.dumps({"passed": False}), encoding="utf-8")
    policy_dir = run_dir / "policy_tables"
    policy_dir.mkdir(exist_ok=True)
    with (policy_dir / "diagnostic_session27_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "ok", "error"])
        writer.writeheader()
        writer.writerow(
            {
                "section": "experiment",
                "name": "pilot-smoke-measured",
                "ok": "False" if failed_gate else "True",
                "error": summary_error or (
                    "session risk signal gate failed"
                    if failed_gate == "risk"
                    else "session trace semantics gate failed"
                    if failed_gate
                    else ""
                ),
            }
        )


def test_run_batches_continues_after_complete_gate_failure_and_merges(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    calls: list[str] = []

    def fake_runner(item: dict[str, object]) -> int:
        calls.append(str(item["batch_id"]))
        _write_supervisor_output(item, failed_gate="trace" if item["batch_id"] == "batch000" else None)
        return 9 if item["batch_id"] == "batch000" else 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert calls == ["batch000", "batch001"]
    assert exit_code == 0
    assert (root / "merged" / "profile_tables" / "diagnostic_session27_profiles.csv").exists()
    assert (root / "merged" / "session_traces" / "diagnostic_session27_trace.csv").exists()
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["diagnostic_only"] is True
    assert summary["merged"] is True
    assert summary["batches"][0]["gate_only_failure"] is True
    assert summary["batches"][0]["mergeable"] is True


def test_run_batches_writes_status_but_refuses_partial_merge(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    second = manifest["batches"][1]  # type: ignore[index]
    (Path(second["run_dir"]) / "profile_tables" / "diagnostic_session27_profiles.csv").unlink()
    calls: list[str] = []

    def fake_runner(item: dict[str, object]) -> int:
        calls.append(str(item["batch_id"]))
        if item["batch_id"] == "batch000":
            _write_supervisor_output(item)
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert calls == ["batch000", "batch001"]
    assert exit_code == 1
    assert not (root / "merged").exists()
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["diagnostic_only"] is True
    assert summary["merged"] is False
    assert summary["batches"][1]["mergeable"] is False


def test_run_batches_removes_stale_merged_output_before_nonmergeable_run(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    second = manifest["batches"][1]  # type: ignore[index]
    (Path(second["run_dir"]) / "profile_tables" / "diagnostic_session27_profiles.csv").unlink()
    stale = root / "merged" / "profile_tables" / "stale.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")

    def fake_runner(item: dict[str, object]) -> int:
        if item["batch_id"] == "batch000":
            _write_supervisor_output(item)
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert exit_code == 1
    assert not (root / "merged").exists()


def test_run_batches_ignores_stale_gate_and_profile_artifacts_before_child_launch(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    first = manifest["batches"][0]  # type: ignore[index]
    first_run_dir = Path(first["run_dir"])
    (first_run_dir / "profile_tables" / "pilot_session_trace_semantics_gate.json").write_text(
        json.dumps({"passed": False}), encoding="utf-8"
    )
    calls: list[str] = []

    def fake_runner(item: dict[str, object]) -> int:
        calls.append(str(item["batch_id"]))
        return 9 if item["batch_id"] == "batch000" else 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert calls == ["batch000", "batch001"]
    assert exit_code == 1
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["batches"][0]["gate_only_failure"] is False
    assert summary["batches"][0]["profile_csv"] is None


def test_run_batches_persists_runner_exception_and_continues(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    calls: list[str] = []

    def fake_runner(item: dict[str, object]) -> int:
        calls.append(str(item["batch_id"]))
        if item["batch_id"] == "batch000":
            raise RuntimeError("child launch exploded")
        _write_supervisor_output(item)
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert calls == ["batch000", "batch001"]
    assert exit_code == 1
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["batches"][0]["mergeable"] is False
    assert "child launch exploded" in summary["batches"][0]["execution_error"]


def test_run_batches_rejects_oom_summary_despite_complete_gate_artifacts(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)

    def fake_runner(item: dict[str, object]) -> int:
        if item["batch_id"] == "batch000":
            _write_supervisor_output(item, failed_gate="trace", summary_error="CUDA out of memory during policy replay")
            return 9
        _write_supervisor_output(item)
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert exit_code == 1
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["batches"][0]["gate_only_failure"] is False
    assert summary["batches"][0]["mergeable"] is False


def test_main_uses_absolute_batch_run_dir_for_relative_run_root(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": request_id,
                    "session_id": "s0",
                    "arrival_index": index,
                    "metadata": {"diagnostic_only": True},
                }
            )
            + "\n"
            for index, request_id in enumerate(("r1", "r2"))
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
profiles:
  names: [full_gpu, lossy]
policies:
  names: [full_lru]
outputs:
  smoke_profiles: out/profile_tables/diagnostic_session27_profiles.csv
  smoke_session_trace: out/session_traces/diagnostic_session27_trace.csv
  smoke_summary: out/policy_tables/diagnostic_session27_summary.csv
""",
        encoding="utf-8",
    )
    relative_root = "diagnostic-runs"
    expected_root = (tmp_path / relative_root).resolve()
    commands: list[list[str]] = []

    def fake_subprocess_run(command: list[str], cwd: Path) -> SimpleNamespace:
        commands.append(command)
        run_dir = Path(command[command.index("--run-dir") + 1])
        config_path = Path(command[command.index("--config") + 1])
        assert run_dir == expected_root / "batch_outputs" / "batch000"
        written_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert written_config["outputs"]["smoke_profiles"] == "out/profile_tables/diagnostic_session27_profiles_batch000.csv"
        _write_supervisor_output(
            {
                "run_dir": str(run_dir),
                "fixture": str(expected_root / "fixtures" / "batch000.jsonl"),
            }
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(diagnostic_runner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_diagnostic_session_batches.py",
            "--fixture",
            str(fixture),
            "--config",
            str(config),
            "--run-root",
            relative_root,
            "--sessions-per-batch",
            "1",
        ],
    )

    assert diagnostic_runner.main() == 0
    assert len(commands) == 1
    assert (expected_root / "merged" / "profile_tables" / "diagnostic_session27_profiles.csv").exists()


def test_run_batches_rejects_schema_drift_without_publishing_partial_merge(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)

    def fake_runner(item: dict[str, object]) -> int:
        _write_supervisor_output(
            item,
            profile_fieldnames=["request_id", "profile", "ok", "measured", "extra"]
            if item["batch_id"] == "batch001"
            else None,
        )
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert exit_code == 1
    assert not (root / "merged").exists()
    assert not (root / ".merged.tmp").exists()
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert "merge_error" in summary


def test_run_batches_rejects_malformed_profile_row_without_partial_merge(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)

    def fake_runner(item: dict[str, object]) -> int:
        _write_supervisor_output(item, malformed_profile_row=item["batch_id"] == "batch001")
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert exit_code == 1
    assert not (root / "merged").exists()


def test_run_batches_records_zero_exit_failed_risk_gate_separately(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)

    def fake_runner(item: dict[str, object]) -> int:
        _write_supervisor_output(item, failed_gate="risk" if item["batch_id"] == "batch000" else None)
        return 0

    exit_code = run_batches(manifest, root, tmp_path, "test", fake_runner)

    assert exit_code == 0
    summary = json.loads((root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    assert summary["batches"][0]["diagnostic_gate_failed"] is True
    assert summary["batches"][0]["gate_only_failure"] is False


def test_materialize_session_batches_merges_final_singleton_without_shifting_batch007(tmp_path: Path) -> None:
    rows = [
        {
            "request_id": f"s{session:02d}t{turn}",
            "session_id": f"s{session:02d}",
            "turn_index": turn,
            "arrival_index": arrival,
            "task": "chat",
            "prompt": "p",
            "reference": "r",
            "metadata": {"diagnostic_only": True},
        }
        for arrival, (turn, session) in enumerate(
            (turn, session) for turn in range(5) for session in range(27)
        )
    ]
    fixture = tmp_path / "session27.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    batches = materialize_session_batches(fixture, tmp_path / "batches", sessions_per_batch=2)

    assert [batch.batch_id for batch in batches] == [f"batch{index:03d}" for index in range(13)]
    assert [batch.session_count for batch in batches] == [2] * 12 + [3]
    assert [batch.request_count for batch in batches] == [10] * 12 + [15]

    batch007_rows = [
        json.loads(line)
        for line in batches[7].fixture_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["session_id"] for row in batch007_rows} == {"s14", "s15"}

    all_session_ids = [
        row["session_id"]
        for batch in batches
        for row in (
            json.loads(line)
            for line in batch.fixture_path.read_text(encoding="utf-8").splitlines()
        )
        if row["turn_index"] == 0
    ]
    assert all_session_ids == [f"s{session:02d}" for session in range(27)]


def test_select_manifest_batches_skips_requested_batch() -> None:
    items = [{"batch_id": "batch006"}, {"batch_id": "batch007"}, {"batch_id": "batch008"}]

    selected = diagnostic_runner.select_manifest_batches(items, skip_batches={"batch007"})

    assert [item["batch_id"] for item in selected] == ["batch006", "batch008"]


def test_select_manifest_batches_runs_only_requested_batch() -> None:
    items = [{"batch_id": "batch006"}, {"batch_id": "batch007"}, {"batch_id": "batch008"}]

    selected = diagnostic_runner.select_manifest_batches(items, only_batches={"batch007"})

    assert [item["batch_id"] for item in selected] == ["batch007"]


def test_select_manifest_batches_rejects_mutually_exclusive_filters() -> None:
    items = [{"batch_id": "batch007"}]

    with pytest.raises(ValueError, match="mutually exclusive"):
        diagnostic_runner.select_manifest_batches(
            items,
            skip_batches={"batch007"},
            only_batches={"batch007"},
        )


def test_select_manifest_batches_rejects_unknown_id_before_selection() -> None:
    items = [{"batch_id": "batch007"}]

    with pytest.raises(ValueError, match="unknown batch IDs"):
        diagnostic_runner.select_manifest_batches(items, only_batches={"batch999"})


def test_run_batches_partial_execution_preserves_unselected_output_and_skips_merge(tmp_path: Path) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    unselected = Path(manifest["batches"][1]["run_dir"])
    marker = unselected / "preserve.marker"
    marker.write_text("keep", encoding="utf-8")
    selected_items = diagnostic_runner.select_manifest_batches(
        manifest["batches"], only_batches={"batch000"}
    )
    executed: list[str] = []

    def fake_runner(item: dict[str, object]) -> int:
        executed.append(str(item["batch_id"]))
        _write_supervisor_output(item)
        return 0

    exit_code = run_batches(
        manifest,
        root,
        tmp_path,
        "test",
        fake_runner,
        selected_batches=selected_items,
    )

    assert exit_code == 0
    assert executed == ["batch000"]
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (root / "merged").exists()


def test_main_only_batches_runs_only_selected_batch(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    rows = [
        {
            "request_id": f"r{session}",
            "session_id": f"s{session}",
            "arrival_index": session,
            "metadata": {"diagnostic_only": True},
        }
        for session in range(4)
    ]
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("profiles:\n  names: [full_gpu, lossy]\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run_batches(manifest, root, repo_root, conda_env, runner, *, selected_batches=None):
        observed["selected"] = [item["batch_id"] for item in selected_batches]
        return 0

    monkeypatch.setattr(diagnostic_runner, "run_batches", fake_run_batches)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_diagnostic_session_batches.py",
            "--fixture",
            str(fixture),
            "--config",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--sessions-per-batch",
            "2",
            "--only-batches",
            "batch001",
        ],
    )

    assert diagnostic_runner.main() == 0
    assert observed["selected"] == ["batch001"]



def test_main_batch_overrides_applies_matching_batch_config(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    rows = [
        {"request_id": f"r{index}", "session_id": f"s{index}", "arrival_index": index}
        for index in range(9)
    ]
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "profile_smoke:\n  device_strategy: balanced_two_gpu\n"
        "outputs:\n  smoke_profiles: out/profile_tables/profiles.csv\n",
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "batch007:\n"
        "  profile_smoke:\n"
        "    device_strategy: balanced_three_gpu\n"
        "    cuda_visible_devices: '0,1,2'\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"

    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(run_root),
        "--sessions-per-batch", "1",
        "--batch-overrides", str(overrides),
        "--prepare-only",
    ])

    assert diagnostic_runner.main() == 0
    written = yaml.safe_load((run_root / "configs" / "batch007.yaml").read_text(encoding="utf-8"))
    assert written["profile_smoke"] == {
        "device_strategy": "balanced_three_gpu",
        "cuda_visible_devices": "0,1,2",
    }


def test_main_batch_overrides_rejects_unknown_batch_before_execution(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps({"request_id": "r0", "session_id": "s0", "arrival_index": 0}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("profiles:\n  names: [full_gpu]\n", encoding="utf-8")
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "batch999:\n  profile_smoke:\n    device_strategy: balanced_three_gpu\n",
        encoding="utf-8",
    )
    called = False

    def fake_run_batches(*args, **kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(diagnostic_runner, "run_batches", fake_run_batches)
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(tmp_path / "run"),
        "--batch-overrides", str(overrides),
    ])

    with pytest.raises(ValueError, match="batch999"):
        diagnostic_runner.main()
    assert called is False

def test_write_batch_config_recursively_merges_override_and_preserves_provenance(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures" / "batch007.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}\n", encoding="utf-8")
    batch = SessionBatch("batch007", fixture, 2, 10)
    config_path = tmp_path / "configs" / "batch007.yaml"
    config_path.parent.mkdir(parents=True)
    run_dir = tmp_path / "batch_outputs" / "batch007"
    base = {
        "runtime": {"device_strategy": "balanced_two_gpu", "nested": {"keep": 1, "replace": 2}},
        "data": {"requests": "base.jsonl"},
        "outputs": {"profile_table": "base_profiles.csv"},
        "run_dir": "base-run",
    }
    override = {
        "runtime": {"nested": {"replace": 9}, "cuda_visible_devices": "0,1,2"},
        "data": {"requests": "override.jsonl"},
        "outputs": {"profile_table": "override.csv"},
        "run_dir": "override-run",
    }

    diagnostic_runner._write_batch_config(
        base,
        batch,
        config_path,
        run_dir,
        override=override,
    )

    written = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert written["runtime"] == {
        "device_strategy": "balanced_two_gpu",
        "nested": {"keep": 1, "replace": 9},
        "cuda_visible_devices": "0,1,2",
    }
    assert written["data"]["requests"] == str(fixture.resolve())
    assert written["data"]["max_requests"] == 10
    assert written["run_dir"] == str(run_dir.resolve())
    assert written["outputs"]["profile_table"] != "override.csv"
    assert "batch007" in written["outputs"]["profile_table"]


def test_main_validate_existing_is_read_only_and_skips_execution(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps({"request_id": "r0", "session_id": "s0", "arrival_index": 0}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("profiles:\n  names: [full_gpu]\n", encoding="utf-8")
    run_root = tmp_path / "run"
    called = False

    def fake_run_batches(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("validate-existing must not execute GPU batches")

    monkeypatch.setattr(diagnostic_runner, "run_batches", fake_run_batches)
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(run_root),
        "--sessions-per-batch", "1",
        "--validate-existing",
    ])

    diagnostic_runner.main()
    assert called is False



def test_main_validate_existing_rejects_missing_batch_output_without_execution(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps({"request_id": "r0", "session_id": "s0", "arrival_index": 0}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("profiles:\n  names: [full_gpu]\n", encoding="utf-8")
    run_root = tmp_path / "run"

    def fake_run_batches(*args, **kwargs):
        raise AssertionError("validate-existing must not execute GPU batches")

    monkeypatch.setattr(diagnostic_runner, "run_batches", fake_run_batches)
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(run_root),
        "--sessions-per-batch", "1",
        "--validate-existing",
    ])

    assert diagnostic_runner.main() != 0



def test_main_validate_existing_does_not_modify_existing_files(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    fixture_dir = run_root / "fixtures"
    config_dir = run_root / "configs"
    output_dir = run_root / "batch_outputs" / "batch000"
    fixture_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    batch_fixture = fixture_dir / "batch000.jsonl"
    batch_fixture.write_text(
        json.dumps({"request_id": "r0", "session_id": "s0", "arrival_index": 0}) + "\n",
        encoding="utf-8",
    )
    batch_config = config_dir / "batch000.yaml"
    batch_config.write_text("profiles:\n  names: [full_gpu]\n", encoding="utf-8")
    manifest = {
        "diagnostic_only": True,
        "sessions_per_batch": 1,
        "profile_names": ["full_gpu"],
        "batches": [{
            "batch_id": "batch000",
            "fixture": str(batch_fixture),
            "config": str(batch_config),
            "run_dir": str(output_dir),
            "sessions": 1,
            "requests": 1,
        }],
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_fixture = tmp_path / "source.jsonl"
    source_fixture.write_text(
        json.dumps({"request_id": "changed", "session_id": "changed", "arrival_index": 99}) + "\n",
        encoding="utf-8",
    )
    source_config = tmp_path / "source.yaml"
    source_config.write_text("profiles:\n  names: [other_profile]\n", encoding="utf-8")

    before = {
        path.relative_to(run_root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_root.rglob("*")
        if path.is_file()
    }

    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(source_fixture),
        "--config", str(source_config),
        "--run-root", str(run_root),
        "--validate-existing",
    ])

    assert diagnostic_runner.main() != 0
    after = {
        path.relative_to(run_root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert after == before



def test_validate_existing_merges_complete_outputs_with_diagnostic_provenance(tmp_path: Path, monkeypatch) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    for item in manifest["batches"]:
        _write_supervisor_output(item)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fixture = tmp_path / "unused.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "unused.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(root),
        "--validate-existing",
    ])

    assert diagnostic_runner.main() == 0
    merged = root / "merged"
    profile_csv = merged / "profile_tables" / "diagnostic_session27_profiles.csv"
    assert profile_csv.is_file()
    merged_manifest = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))
    assert merged_manifest["mixed_hardware"] is True
    assert merged_manifest["performance_comparability"] == "diagnostic_only"


def test_validate_existing_rejects_incomplete_global_session_coverage(tmp_path: Path, monkeypatch) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    manifest["expected_sessions"] = 27
    manifest["expected_requests"] = 135
    manifest["expected_profiles"] = 8
    manifest["expected_profile_rows"] = 1080
    for item in manifest["batches"]:
        _write_supervisor_output(item)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fixture = tmp_path / "unused.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "unused.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(root),
        "--validate-existing",
    ])

    assert diagnostic_runner.main() != 0
    assert not (root / "merged").exists()


def test_validate_batch_output_rejects_dry_run_and_failed_chunk_residue(tmp_path: Path) -> None:
    rows = [
        {
            "request_id": request_id,
            "profile": profile,
            "ok": "True",
            "measured": "True",
            "dry_run": "False",
        }
        for request_id in ("r1", "r2")
        for profile in ("full_gpu", "lossy")
    ]
    rows[0]["dry_run"] = "True"
    batch, run_dir = _batch_output(tmp_path, rows)
    (run_dir / "profile_tables" / "diagnostic_session27_profiles_failed_chunks.csv").write_text(
        "request_id,profile,error\nr1,full_gpu,CUDA error\n",
        encoding="utf-8",
    )

    status = validate_batch_output(batch, run_dir, {"full_gpu", "lossy"})

    assert status["mergeable"] is False
    assert any("dry-run" in error for error in status["errors"])
    assert any("failed chunk" in error for error in status["errors"])


def test_validate_existing_rejects_duplicate_canonical_request_profile_pair(tmp_path: Path, monkeypatch) -> None:
    manifest, root = _supervisor_manifest(tmp_path)
    for item in manifest["batches"]:
        _write_supervisor_output(item)
    second = manifest["batches"][1]
    second_fixture = Path(second["fixture"])
    rows = [json.loads(line) for line in second_fixture.read_text(encoding="utf-8").splitlines()]
    rows[0]["request_id"] = "r0a"
    second_fixture.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_supervisor_output(second)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fixture = tmp_path / "unused.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "unused.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_diagnostic_session_batches.py",
        "--fixture", str(fixture),
        "--config", str(config),
        "--run-root", str(root),
        "--validate-existing",
    ])

    assert diagnostic_runner.main() != 0
    assert not (root / "merged").exists()
