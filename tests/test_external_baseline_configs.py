from __future__ import annotations

from pathlib import Path

import yaml


def test_external_baseline_quality_config_points_to_imported_fixture() -> None:
    config = yaml.safe_load(Path("configs/pilot_external_baseline_quality.yaml").read_text(encoding="utf-8"))

    assert config["experiment"]["type"] == "baseline_quality"
    assert config["data"]["source"] == "fixture"
    assert config["data"]["quality_mode"] == "baseline"
    assert config["data"]["requests"] == "data/fixtures/baseline_quality_external.jsonl"
    assert config["data"]["max_requests"] == 180


def test_external_baseline_session_config_points_to_imported_fixture() -> None:
    config = yaml.safe_load(Path("configs/pilot_external_baseline_session.yaml").read_text(encoding="utf-8"))

    assert config["experiment"]["type"] == "baseline_session"
    assert config["data"]["source"] == "fixture"
    assert config["data"]["quality_mode"] == "session_diagnostic"
    assert config["data"]["requests"] == "data/fixtures/baseline_session_external.jsonl"
    assert config["data"]["max_requests"] == 240
    assert config["outputs"]["smoke_trace_semantics_gate"].endswith("_trace_semantics_gate.json")
    assert config["outputs"]["smoke_risk_signal_gate"].endswith("_risk_signal_gate.json")


def test_run_external_baseline_smoke_script_references_new_configs() -> None:
    script = Path("scripts/run_external_baseline_smoke.sh").read_text(encoding="utf-8")

    assert "pilot_external_baseline_quality.yaml" in script
    assert "pilot_external_baseline_session.yaml" in script
    assert "run_pilot_measured_async.sh" in script
    assert "pilot-smoke-measured" not in script


def test_async_launcher_uses_live_conda_output() -> None:
    script = Path("scripts/run_pilot_measured_async.sh").read_text(encoding="utf-8")

    assert "--no-capture-output" in script or "--live-stream" in script
