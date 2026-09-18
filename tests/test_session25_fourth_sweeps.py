from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from run_util.core_types import ProfileMeasurement
from scripts.run_session25_fourth_sweeps import (
    POLICIES,
    POLICY_OUTPUT_HEADER,
    _record_seed,
    _request_split_audit,
    _validate_policy_csv,
    _assert_attempt_output_available,
    cells,
    run,
)


def test_grid_has_one_seed_and_four_constraint_settings_per_budget() -> None:
    budgets = [15, 20, 25, 30, 35, 50, 64, 80, 96, 120.2578125, 469.6015625]

    schedule = cells(budgets)

    assert len(schedule) == 44
    assert len(schedule) * len(POLICIES) == 220
    assert {budget for _, budget, _, _ in schedule} == set(budgets)
    assert len(set(schedule)) == 44


def test_run_resumes_high_budget_csvs_and_executes_only_new_low_budget_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_root = tmp_path / "profile-run"
    profile_root.joinpath("merged", "profile_tables").mkdir(parents=True)
    (profile_root / "merged" / "profile_tables" / "session25_profiles.csv").write_text("placeholder\n", encoding="utf-8")
    (profile_root / "manifest.json").write_text(
        json.dumps({
            "expected_sessions": 25,
            "expected_requests": 125,
            "expected_profile_rows": 1000,
            "diagnostic_only": True,
            "filtered_fixture": str(profile_root / "filtered_fixture.jsonl"),
        }),
        encoding="utf-8",
    )
    (profile_root / "supervisor_manifest.json").write_text('{"merged": true}', encoding="utf-8")
    (profile_root / "budgets.json").write_text(
        json.dumps({"diagnostic_only": True, "memory_budgets_mib": [120.2578125, 469.6015625]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    budgets = [15, 20, 25, 30, 35, 50, 64, 80, 96, 120.2578125, 469.6015625]
    config_path.write_text(
        yaml.safe_dump({
            "diagnostic_only": True,
            "policies": {"backend": "online_qwen"},
            "pilot": {"memory_budgets_mib": budgets},
            "data": {"requests": "fixture.jsonl", "calibration_fraction": 0.5},
        }),
        encoding="utf-8",
    )
    attempt_root = tmp_path / "attempt"
    high_budget_suffixes = {"mem120p258.csv", "mem469p602.csv"}
    commands: list[list[str]] = []

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.audit_fixture", lambda _: {"fixture": "ok"})
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.read_measurements", lambda _: [object()])
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps._request_split_audit",
        lambda *_args, **_kwargs: {
            "total_request_count": 125,
            "calibration_request_count": 60,
            "evaluation_request_count": 65,
            "evaluation_request_ids": [f"r{index}" for index in range(65)],
        },
    )
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps._assert_attempt_output_available",
        lambda path, *_: any(path.name.endswith(suffix) for suffix in high_budget_suffixes),
    )

    def fake_command(command: list[str], log_path: Path) -> int:
        commands.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("complete\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._command", fake_command)
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._record_seed", lambda *_: None)
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._validate_policy_csv", lambda *_: True)
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.aggregate_session27", lambda *_args, **_kwargs: None)

    assert run(Namespace(run_root=profile_root, attempt_root=attempt_root, config=config_path, conda_env="test")) == 0

    assert len(commands) == 36
    observed_budgets = {float(command[command.index("--memory-budget-mib") + 1]) for command in commands}
    assert observed_budgets == {15, 20, 25, 30, 35, 50, 64, 80, 96}
    assert {(float(command[command.index("--epsilon") + 1]), float(command[command.index("--delta") + 1])) for command in commands} == {
        (0.05, 0.05), (0.05, 0.1), (0.1, 0.05), (0.1, 0.1)
    }
    status = json.loads((attempt_root / "fourth_grid_status.json").read_text(encoding="utf-8"))
    assert status["planned_cell_configurations"] == 44
    assert status["planned_cells"] == 220
    assert [cell["state"] for cell in status["cells"]].count("resumed_success") == 8
    assert [cell["state"] for cell in status["cells"]].count("success") == 36


def test_seed_provenance_and_complete_policy_rows(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_id", "policy", "diagnostic_only", "backend_name", "ok"])
        writer.writeheader()
        writer.writerows(
            {
                "request_id": f"r{index}",
                "policy": policy,
                "diagnostic_only": "True",
                "backend_name": "online_qwen",
                "ok": "True",
            }
            for policy in POLICIES for index in range(65)
        )
    expected_ids = {f"r{index}" for index in range(65)}
    assert not _validate_policy_csv(path, 20260906, expected_ids)
    _record_seed(path, 20260906)
    assert _validate_policy_csv(path, 20260906, expected_ids)
    assert not _validate_policy_csv(path, 20260907, expected_ids)


def test_attempt_output_recovers_complete_legacy_csv_missing_only_split_seed(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    expected_ids = {f"r{index}" for index in range(65)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field for field in POLICY_OUTPUT_HEADER if field != "split_seed"])
        writer.writeheader()
        writer.writerows(
            {
                "request_id": request_id,
                "policy": policy,
                "diagnostic_only": "True",
                "backend_name": "online_qwen",
                "ok": "True",
            }
            for policy in POLICIES for request_id in sorted(expected_ids)
        )

    assert _assert_attempt_output_available(path, 20260906, expected_ids)
    with path.open(encoding="utf-8", newline="") as handle:
        assert "split_seed" in (csv.DictReader(handle).fieldnames or [])


def test_attempt_output_rejects_legacy_csv_missing_standard_output_column(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    expected_ids = {f"r{index}" for index in range(65)}
    fieldnames = [field for field in POLICY_OUTPUT_HEADER if field not in {"action_profile", "split_seed"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "request_id": request_id,
                "policy": policy,
                "diagnostic_only": "True",
                "backend_name": "online_qwen",
                "ok": "True",
            }
            for policy in POLICIES for request_id in sorted(expected_ids)
        )
    original_header = path.read_text(encoding="utf-8").splitlines()[0]

    with pytest.raises(ValueError, match="existing incomplete policy CSV"):
        _assert_attempt_output_available(path, 20260906, expected_ids)

    recovered_header = path.read_text(encoding="utf-8").splitlines()[0]
    assert recovered_header == original_header
    assert "split_seed" not in recovered_header.split(",")


def test_attempt_output_rejects_legacy_csv_with_extra_header_column(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    expected_ids = {f"r{index}" for index in range(65)}
    fieldnames = [field for field in POLICY_OUTPUT_HEADER if field != "split_seed"] + ["unexpected_column"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "request_id": request_id,
                "policy": policy,
                "diagnostic_only": "True",
                "backend_name": "online_qwen",
                "ok": "True",
            }
            for policy in POLICIES for request_id in sorted(expected_ids)
        )
    original_header = path.read_text(encoding="utf-8").splitlines()[0]

    with pytest.raises(ValueError, match="existing incomplete policy CSV"):
        _assert_attempt_output_available(path, 20260906, expected_ids)

    recovered_header = path.read_text(encoding="utf-8").splitlines()[0]
    assert recovered_header == original_header
    assert "split_seed" not in recovered_header.split(",")


def test_attempt_output_rejects_legacy_csv_with_reordered_header(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    expected_ids = {f"r{index}" for index in range(65)}
    fieldnames = [field for field in POLICY_OUTPUT_HEADER if field != "split_seed"]
    fieldnames[0], fieldnames[1] = fieldnames[1], fieldnames[0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "request_id": request_id,
                "policy": policy,
                "diagnostic_only": "True",
                "backend_name": "online_qwen",
                "ok": "True",
            }
            for policy in POLICIES for request_id in sorted(expected_ids)
        )
    original_header = path.read_text(encoding="utf-8").splitlines()[0]

    with pytest.raises(ValueError, match="existing incomplete policy CSV"):
        _assert_attempt_output_available(path, 20260906, expected_ids)

    recovered_header = path.read_text(encoding="utf-8").splitlines()[0]
    assert recovered_header == original_header
    assert "split_seed" not in recovered_header.split(",")


def test_request_split_audit_derives_sixty_calibration_and_sixty_five_evaluation_requests() -> None:
    measurements = [
        ProfileMeasurement(
            request_id=f"r{session_index:02d}-{turn_index}",
            session_id=f"s{session_index:02d}",
            turn_index=turn_index,
            profile="full_gpu",
            adapter="full",
            ok=True,
            measured=True,
            extra={"task": "qa", "length_bucket": "short"},
        )
        for session_index in range(25)
        for turn_index in range(5)
    ]

    audit = _request_split_audit(
        measurements,
        split_seed=20260906,
        calibration_fraction=0.5,
        stratify_session=True,
    )

    assert audit["total_request_count"] == 125
    assert audit["calibration_request_count"] == 60
    assert audit["evaluation_request_count"] == 65
    assert len(audit["evaluation_request_ids"]) == 65


def test_policy_csv_rejects_duplicate_evaluation_request_ids(tmp_path: Path) -> None:
    path = tmp_path / "policy.csv"
    expected_ids = {"r1", "r2"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["request_id", "policy", "diagnostic_only", "backend_name", "ok", "split_seed"],
        )
        writer.writeheader()
        for policy in POLICIES:
            writer.writerows(
                {
                    "request_id": "r1",
                    "policy": policy,
                    "diagnostic_only": "True",
                    "backend_name": "online_qwen",
                    "ok": "True",
                    "split_seed": "20260906",
                }
                for _ in range(2)
            )

    assert not _validate_policy_csv(path, 20260906, expected_ids)


def test_run_separates_profile_root_from_attempt_root_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_root = tmp_path / "profile-run"
    attempt_root = profile_root / "policy_attempts" / "retry1"
    profile_root.joinpath("merged", "profile_tables").mkdir(parents=True)
    profile_path = profile_root / "merged" / "profile_tables" / "session25_profiles.csv"
    profile_path.write_text("placeholder\n", encoding="utf-8")
    profile_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "expected_sessions": 25,
                "expected_requests": 125,
                "expected_profile_rows": 1000,
                "diagnostic_only": True,
                "filtered_fixture": str(profile_root / "filtered_fixture.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    profile_root.joinpath("supervisor_manifest.json").write_text('{"merged": true}', encoding="utf-8")
    profile_root.joinpath("budgets.json").write_text(
        json.dumps({"diagnostic_only": True, "memory_budgets_mib": [4900, 5000]}),
        encoding="utf-8",
    )
    old_path = profile_root / "seed20260906" / "policy_tables" / "old-incomplete.csv"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("request_id,policy\nr0,full_lru\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "diagnostic_only: true\npolicies:\n  backend: online_qwen\ndata:\n  requests: fixture.jsonl\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")

    measurements = [
        ProfileMeasurement(
            request_id=f"r{session_index:02d}-{turn_index}",
            session_id=f"s{session_index:02d}",
            turn_index=turn_index,
            profile="full_gpu",
            adapter="full",
            ok=True,
            measured=True,
            extra={"task": "qa", "length_bucket": "short"},
        )
        for session_index in range(25)
        for turn_index in range(5)
    ]

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.audit_fixture", lambda _: {"fixture": "ok"})
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.read_measurements", lambda _: measurements)
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps._request_split_audit",
        lambda *_args, **_kwargs: {
            "total_request_count": 125,
            "calibration_request_count": 60,
            "evaluation_request_count": 65,
            "evaluation_request_ids": [f"r{index}" for index in range(65)],
        },
    )

    def fake_command(command: list[str], log_path: Path) -> int:
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["request_id", "policy", "diagnostic_only", "backend_name", "ok"],
            )
            writer.writeheader()
            for policy in POLICIES:
                for index in range(65):
                    writer.writerow(
                        {
                            "request_id": f"r{index}",
                            "policy": policy,
                            "diagnostic_only": "True",
                            "backend_name": "online_qwen",
                            "ok": "True",
                        }
                    )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._command", fake_command)
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.aggregate_session27", lambda *_args, **_kwargs: None)

    result = run(Namespace(run_root=profile_root, attempt_root=attempt_root, config=config_path, conda_env="test", preflight=False))

    assert result == 0
    assert not (profile_root / "seed20260906" / "config.yaml").exists()
    assert not (profile_root / "fourth_grid_status.json").exists()
    assert (attempt_root / "seed20260906" / "config.yaml").exists()
    assert (attempt_root / "fourth_grid_status.json").exists()
    status = json.loads((attempt_root / "fourth_grid_status.json").read_text(encoding="utf-8"))
    assert status["profile_run_root"] == str(profile_root)
    assert status["attempt_root"] == str(attempt_root)
    assert status["fixture"] == {"fixture": "ok"}
    assert status["total_request_count"] == 125
    assert status["calibration_request_count"] == 60
    assert status["evaluation_request_count"] == 65
    assert status["planned_cells"] == 40
    assert all(Path(cell["output"]).is_relative_to(attempt_root) for cell in status["cells"])


def test_run_rejects_incomplete_csv_in_attempt_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempt_root = tmp_path / "attempt"
    path = attempt_root / "seed20260906" / "policy_tables" / "session25_policy_eps0p05_delta0p05_mem100.csv"
    path.parent.mkdir(parents=True)
    path.write_text("request_id,policy\nr0,full_lru\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing incomplete policy CSV"):
        _assert_attempt_output_available(path, 20260906, {"r0"})


def test_run_continues_after_failed_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_root = Path("out/full_25session_baseline_9_14/full").resolve()
    attempt_root = tmp_path / "attempt"
    config_path = Path("configs/pilot_diagnostic_session25_fourth.yaml").resolve()
    cells_to_run = [
        (20260906, 120.2578125, 0.05, 0.05),
        (20260906, 120.2578125, 0.05, 0.1),
    ]
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.cells", lambda _: cells_to_run)
    calls: list[Path] = []

    def fake_command(command: list[str], log_path: Path) -> int:
        output_path = Path(command[command.index("--output") + 1])
        calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("incomplete\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("failure\n", encoding="utf-8")
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._command", fake_command)
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps._validate_policy_csv",
        lambda path, *_: len(calls) > 1 and path == calls[-1],
    )
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._record_seed", lambda *_: None)
    aggregate_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps.aggregate_session27",
        lambda input_dirs, output_root, **kwargs: aggregate_calls.append({
            "input_dirs": input_dirs, "output_root": output_root, **kwargs,
        }),
    )

    result = run(Namespace(run_root=profile_root, attempt_root=attempt_root, config=config_path, conda_env="test"))

    assert result == 1
    assert len(calls) == 2
    status = json.loads((attempt_root / "fourth_grid_status.json").read_text(encoding="utf-8"))
    assert status["finished_cells"] == 10
    assert [cell["state"] for cell in status["cells"]] == ["failed", "success"]
    assert len(aggregate_calls) == 1
    assert aggregate_calls[0]["input_paths"] == [calls[1]]


@pytest.mark.parametrize("target_state", ["missing", "invalid"])
def test_run_resume_only_rejects_unavailable_policy_csv_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_state: str
) -> None:
    run_root = tmp_path / "profile-run"
    attempt_root = tmp_path / "attempt"
    run_root.joinpath("merged", "profile_tables").mkdir(parents=True)
    (run_root / "merged" / "profile_tables" / "session25_profiles.csv").write_text(
        "placeholder\n", encoding="utf-8"
    )
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "expected_sessions": 25,
                "expected_requests": 125,
                "expected_profile_rows": 1000,
                "diagnostic_only": True,
                "filtered_fixture": str(run_root / "filtered_fixture.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    (run_root / "supervisor_manifest.json").write_text('{"merged": true}', encoding="utf-8")
    (run_root / "budgets.json").write_text(
        json.dumps({"diagnostic_only": True, "memory_budgets_mib": [100, 200]}), encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "diagnostic_only: true\npolicies:\n  backend: online_qwen\ndata:\n  requests: fixture.jsonl\n",
        encoding="utf-8",
    )
    target = attempt_root / "seed20260906" / "policy_tables" / "session25_policy_eps0p05_delta0p05_mem100.csv"
    if target_state == "invalid":
        target.parent.mkdir(parents=True)
        target.write_text("request_id,policy\nr0,full_lru\n", encoding="utf-8")

    measurements = [object()]
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.audit_fixture", lambda _: {"fixture": "ok"})
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.read_measurements", lambda _: measurements)
    monkeypatch.setattr(
        "scripts.run_session25_fourth_sweeps._request_split_audit",
        lambda *_args, **_kwargs: {
            "total_request_count": 1,
            "calibration_request_count": 0,
            "evaluation_request_count": 1,
            "evaluation_request_ids": ["r0"],
        },
    )
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.cells", lambda _: [(20260906, 100.0, 0.05, 0.05)])
    command_calls: list[object] = []

    def unexpected_command(*args: object, **kwargs: object) -> int:
        command_calls.append((args, kwargs))
        raise AssertionError("resume-only must not execute _command")

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._command", unexpected_command)

    with pytest.raises(ValueError, match="resume-only.*policy CSV"):
        run(
            Namespace(
                run_root=run_root,
                attempt_root=attempt_root,
                config=config_path,
                conda_env="test",
                resume_only=True,
            )
        )

    assert command_calls == []


def test_run_resume_only_rejects_missing_budgets_before_any_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "profile-run"
    run_root.joinpath("merged", "profile_tables").mkdir(parents=True)
    (run_root / "merged" / "profile_tables" / "session25_profiles.csv").write_text(
        "placeholder\n", encoding="utf-8"
    )
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "expected_sessions": 25,
                "expected_requests": 125,
                "expected_profile_rows": 1000,
                "diagnostic_only": True,
                "filtered_fixture": str(run_root / "filtered_fixture.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    (run_root / "supervisor_manifest.json").write_text('{"merged": true}', encoding="utf-8")
    target = run_root / "seed20260906" / "policy_tables" / "session25_policy_eps0p05_delta0p05_mem100.csv"
    target.parent.mkdir(parents=True)
    target.write_text("request_id,policy\nr0,full_lru\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "diagnostic_only: true\npolicies:\n  backend: online_qwen\ndata:\n  requests: fixture.jsonl\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.audit_fixture", lambda _: {"fixture": "ok"})
    monkeypatch.setattr("scripts.run_session25_fourth_sweeps.read_measurements", lambda _: [object()])
    command_calls: list[object] = []

    def unexpected_command(*args: object, **kwargs: object) -> int:
        command_calls.append((args, kwargs))
        raise AssertionError("resume-only must not derive budgets or execute policy command")

    monkeypatch.setattr("scripts.run_session25_fourth_sweeps._command", unexpected_command)

    with pytest.raises(ValueError, match="resume-only.*budgets file"):
        run(
            Namespace(
                run_root=run_root,
                attempt_root=run_root,
                config=config_path,
                conda_env="test",
                resume_only=True,
            )
        )

    assert command_calls == []
