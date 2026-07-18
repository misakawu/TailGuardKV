from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from calibration.conformal import ConformalGuard
from core_types import Action
from core_types import PolicyRunRecord, ProfileMeasurement, ProfileSpec, Request
from experiment_common import config_profiles, load_config, validate_profile_measurements, with_quality, write_csv
from metrics import MetricCollector
from policies.base import Policy
from policies.registry import build_policies
from policies.tailguard import TailGuardPolicy
from profiles.base import qwen2_kv_profile_measurement
from run_build_profile_table import build_profile_table
from run_experiment import run_policies


def _measurement(
    request_id: str,
    profile: str,
    quality_loss: float | None,
    ttft_ms: float = 10.0,
    peak_memory_mib: float = 100.0,
    measured: bool = True,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        profile=profile,
        adapter="full",
        ok=True,
        measured=measured,
        output_text=f"{request_id}:{profile}",
        ttft_ms=ttft_ms,
        peak_memory_mib=peak_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=quality_loss,
        extra={"task": "qa", "length_bucket": "short", "split": "calibration"},
    )


class TailGuardCoreTest(unittest.TestCase):
    def test_pytest_ini_limits_collection_to_project_tests(self) -> None:
        pytest_ini = Path("pytest.ini").read_text(encoding="utf-8")
        self.assertIn("testpaths = tests", pytest_ini)
        self.assertIn("third_party", pytest_ini)

    def test_conformal_guard_uses_bonferroni_delta_and_residual_quantile(self) -> None:
        guard = ConformalGuard(
            epsilon=0.5,
            delta=0.2,
            calibration_rows=[
                _measurement("r1", "kivi_4bit", 0.1),
                _measurement("r2", "kivi_4bit", 0.2),
                _measurement("r3", "kivi_4bit", 0.3),
                _measurement("r4", "kivi_2bit", 0.4),
                _measurement("r5", "kivi_2bit", 0.5),
            ],
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertAlmostEqual(guard.delta_a, 0.1)
        self.assertAlmostEqual(guard.risk_upper(request, "kivi_4bit", 0.1), 0.25)
        self.assertTrue(guard.is_safe(request, "kivi_4bit", 0.1))

    def test_conformal_guard_sparse_groups_are_unsafe_and_exact_safe(self) -> None:
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            calibration_rows=[_measurement("r1", "kivi_4bit", 0.1)],
            exact_profiles={"full_gpu"},
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(guard.risk_upper(request, "full_gpu", 0.9), 0.0)
        self.assertFalse(guard.is_safe(request, "kivi_4bit", 0.0))

    def test_tailguard_falls_back_to_exact_when_no_safe_lossy_candidate(self) -> None:
        rows = [
            _measurement("eval1", "full_gpu", 0.0, ttft_ms=20.0, peak_memory_mib=120.0),
            _measurement("eval1", "kivi_4bit", 0.4, ttft_ms=5.0, peak_memory_mib=60.0),
        ]
        policy = TailGuardPolicy(rows, ["full_gpu", "kivi_4bit"], 0.2, 0.05, {"full_gpu"})
        action = policy.decide(Request("eval1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "full_gpu")
        self.assertTrue(action.safe)

    def test_tailguard_budget_filters_safe_lossy_then_chooses_fastest_p95_ttft(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0, peak_memory_mib=120.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=30.0, peak_memory_mib=120.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=4.0, peak_memory_mib=80.0),
            _measurement("c2", "kivi_4bit", 0.1, ttft_ms=6.0, peak_memory_mib=80.0),
            _measurement("c1", "h2o_heavy_hitter", 0.1, ttft_ms=2.0, peak_memory_mib=200.0),
            _measurement("c2", "h2o_heavy_hitter", 0.1, ttft_ms=3.0, peak_memory_mib=200.0),
        ]
        policy = TailGuardPolicy(
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.2,
            0.05,
            {"full_gpu"},
            memory_budget_mib=100.0,
        )
        action = policy.decide(Request("eval1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "kivi_4bit")
        self.assertTrue(action.safe)

    def test_tailguard_does_not_choose_unsafe_lossy_fallback(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=20.0, peak_memory_mib=120.0),
            _measurement("c1", "kivi_4bit", 0.8, ttft_ms=5.0, peak_memory_mib=60.0),
        ]
        policy = TailGuardPolicy(rows, ["kivi_4bit", "full_gpu"], 0.2, 0.05, {"full_gpu"})
        action = policy.decide(Request("eval1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "full_gpu")
        self.assertIn("exact fallback", action.fallback_reason)
        self.assertTrue(action.safe)

    def test_registry_builds_split_policy_modules(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        policies = build_policies(
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic", "tailguard", "quality_oracle"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(
            [policy.name for policy in policies],
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic", "tailguard", "quality_oracle"],
        )

    def test_worst_group_uses_task_length_bucket_and_profile(self) -> None:
        records = [
            PolicyRunRecord(
                policy="p",
                request_id="r1",
                task="qa",
                length_bucket="short",
                action_profile="kivi_4bit",
                ok=True,
                measured=True,
                reason="same",
                quality_loss=0.3,
            ),
            PolicyRunRecord(
                policy="p",
                request_id="r2",
                task="summary",
                length_bucket="long",
                action_profile="kivi_4bit",
                ok=True,
                measured=True,
                reason="same",
                quality_loss=0.0,
            ),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["worst_group_violation"], 1.0)

    def test_metrics_delta_slack_uses_delta_not_epsilon(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "kivi_4bit", True, True, quality_loss=0.3),
            PolicyRunRecord("p", "r2", "kivi_4bit", True, True, quality_loss=0.0),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["target_delta"], 0.05)
        self.assertEqual(summary["p"]["violation_rate"], 0.5)
        self.assertEqual(summary["p"]["delta_slack"], -0.45)

    def test_validate_profile_measurements_requires_all_configured_profiles(self) -> None:
        rows = [_measurement("r1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "缺少必需 profile"):
            validate_profile_measurements(rows, required_profiles=["full_gpu", "kivi_4bit"])

    def test_validate_profile_measurements_requires_ok_for_measured_replay(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="full_gpu",
            adapter="full",
            ok=False,
            measured=True,
            output_text="x",
            ttft_ms=1.0,
            peak_memory_mib=1.0,
            quality_loss=0.0,
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )
        with self.assertRaisesRegex(ValueError, "ok=True"):
            validate_profile_measurements([row], require_measured=True)

    def test_validate_profile_measurements_rejects_empty_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile 表为空"):
            validate_profile_measurements([])

    def test_validate_profile_measurements_rejects_lossy_without_full_baseline_quality(self) -> None:
        rows = [
            _measurement("r1", "kivi_4bit", None),
        ]
        with self.assertRaisesRegex(ValueError, "quality_loss"):
            validate_profile_measurements(rows, require_measured=True, required_profiles=["kivi_4bit"])

    def test_e0_config_uses_three_reproducible_profiles(self) -> None:
        config = load_config(Path("configs/e0_reproduce.yaml"))
        self.assertEqual(config_profiles(config), ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"])
        self.assertEqual(config["model"]["pilot_model"], config["model"]["profile_smoke_model"])
        self.assertEqual(config["data"]["requests"], "data/fixtures/e0_reproduce_requests.jsonl")

    def test_run_policies_rejects_dry_run_replay_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dry.csv"
            write_csv(
                path,
                [
                    {
                        "request_id": "r1",
                        "profile": "full_gpu",
                        "adapter": "full",
                        "ok": True,
                        "measured": False,
                        "output_text": "x",
                        "error": "",
                        "latency_ms": 1.0,
                        "ttft_ms": 1.0,
                        "peak_memory_mib": 10.0,
                        "resident_memory_mib": 10.0,
                        "quality_score": 1.0,
                        "quality_loss": 0.0,
                        "task": "qa",
                        "length_bucket": "short",
                        "split": "eval",
                    }
                ],
            )
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(path),
                output=str(Path(tmpdir) / "policy.csv"),
                profiles=None,
                policies=["full_lru"],
                epsilon=None,
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            self.assertEqual(run_policies(args), 2)

    def test_build_profile_table_import_bad_table_returns_2_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.csv"
            output_path = Path(tmpdir) / "profiles.csv"
            bad_path.write_text("request_id,profile\nr1,full_gpu\n", encoding="utf-8")
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                adapters=None,
                output=str(output_path),
                import_measurements=str(bad_path),
                dry_run=True,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = build_profile_table(args)
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 2)
            self.assertFalse(payload["ok"])
            self.assertNotIn("Traceback", stream.getvalue())

    def test_run_policies_uses_defaults_when_pilot_thresholds_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  names:",
                        "    - full_gpu",
                        "policies:",
                        "  names:",
                        "    - full_lru",
                    ]
                ),
                encoding="utf-8",
            )
            write_csv(measurements_path, [_measurement("e1", "full_gpu", 0.0).to_row()])
            args = argparse.Namespace(
                config=str(config_path),
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=None,
                policies=None,
                epsilon=None,
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = run_policies(args)
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["epsilon"], 0.2)
            self.assertEqual(payload["delta"], 0.05)

    def test_run_policies_rejects_invalid_numeric_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            write_csv(measurements_path, [_measurement("e1", "full_gpu", 0.0).to_row()])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["full_lru"],
                epsilon="bad",
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = run_policies(args)
            self.assertEqual(code, 2)
            self.assertIn("epsilon", stream.getvalue())

    def test_policy_decide_exception_records_failure_and_continues(self) -> None:
        class BrokenPolicy(Policy):
            name = "broken"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                raise RuntimeError(f"cannot decide {request.request_id}")

        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            rows = [
                _measurement("c1", "full_gpu", 0.0),
                _measurement("e1", "full_gpu", 0.0),
                _measurement("e2", "full_gpu", 0.0),
            ]
            rows = [
                row if row.request_id == "c1" else ProfileMeasurement(
                    request_id=row.request_id,
                    profile=row.profile,
                    adapter=row.adapter,
                    ok=row.ok,
                    measured=row.measured,
                    output_text=row.output_text,
                    error=row.error,
                    latency_ms=row.latency_ms,
                    ttft_ms=row.ttft_ms,
                    peak_memory_mib=row.peak_memory_mib,
                    resident_memory_mib=row.resident_memory_mib,
                    quality_score=row.quality_score,
                    quality_loss=row.quality_loss,
                    extra={**row.extra, "split": "eval"},
                )
                for row in rows
            ]
            write_csv(measurements_path, [row.to_row() for row in rows])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["broken"],
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            with patch("run_run_policies.build_policies", return_value=[BrokenPolicy()]):
                code = run_policies(args)
            self.assertEqual(code, 1)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["ok"] == "False" for record in records))
            self.assertIn("cannot decide e1", records[0]["error"])

    def test_with_quality_uses_any_exact_profile_as_baseline(self) -> None:
        rows = [
            _measurement("r1", "full_cpu", None),
            _measurement("r1", "kivi_4bit", None),
        ]
        updated = with_quality(rows, {"full_cpu"})
        by_profile = {row.profile: row for row in updated}
        self.assertEqual(by_profile["full_cpu"].quality_loss, 0.0)
        self.assertIsNotNone(by_profile["kivi_4bit"].quality_loss)

    def test_tailguard_exact_fallback_record_fields_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            rows = [
                _measurement("c1", "full_gpu", 0.0),
                _measurement("c1", "kivi_4bit", 0.8, ttft_ms=5.0),
                _measurement("e1", "full_gpu", 0.0),
                _measurement("e1", "kivi_4bit", 0.8, ttft_ms=5.0),
            ]
            rows = [
                row if row.request_id == "c1" else ProfileMeasurement(
                    request_id=row.request_id,
                    profile=row.profile,
                    adapter=row.adapter,
                    ok=row.ok,
                    measured=row.measured,
                    output_text=row.output_text,
                    error=row.error,
                    latency_ms=row.latency_ms,
                    ttft_ms=row.ttft_ms,
                    peak_memory_mib=row.peak_memory_mib,
                    resident_memory_mib=row.resident_memory_mib,
                    quality_score=row.quality_score,
                    quality_loss=row.quality_loss,
                    extra={**row.extra, "split": "eval"},
                )
                for row in rows
            ]
            write_csv(measurements_path, [row.to_row() for row in rows])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu", "kivi_4bit"],
                policies=["tailguard"],
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )

            self.assertEqual(run_policies(args), 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                [record] = list(csv.DictReader(handle))
            self.assertEqual(record["action_profile"], "full_gpu")
            self.assertEqual(record["exact"], "True")
            self.assertEqual(record["safe"], "True")
            self.assertEqual(record["task"], "qa")
            self.assertEqual(record["length_bucket"], "short")
            self.assertIn("exact fallback", record["fallback_reason"])

    def test_qwen2_kivi_runtime_proof_missing_is_unmeasured_failure(self) -> None:
        proc = Mock(
            returncode=1,
            stdout='{"ok": false, "measured": false, "error": "KIVI proof missing", "backend": "qwen2_kivi", "kivi_kernel_calls": 0}\n',
            stderr="",
        )
        with patch("profiles.base.subprocess.run", return_value=proc):
            row = qwen2_kv_profile_measurement(
                "kivi",
                "edgekv-kivi",
                Request("r1", "qa", "prompt"),
                ProfileSpec("kivi_4bit", "kivi", "edgekv-kivi", lossy=True, metadata={"bits": 4}),
                {"profile_smoke_model": "/models/qwen-smoke", "pilot_model": "/models/qwen", "max_new_tokens": 1},
            )
        self.assertFalse(row.ok)
        self.assertFalse(row.measured)
        self.assertIn("KIVI proof missing", row.error or "")
        self.assertEqual(str(row.extra["kivi_kernel_calls"]), "0")
        self.assertEqual(row.extra["model"], "/models/qwen-smoke")

    def test_qwen2_h2o_short_request_without_prune_is_unmeasured_failure(self) -> None:
        proc = Mock(
            returncode=1,
            stdout=(
                '{"ok": false, "measured": false, "error": "H2O proof missing: prompt_tokens=8 budget=10 prune_events=0", '
                '"backend": "qwen2_h2o", "h2o_prune_events": 0, "h2o_cache_budget": 10}\n'
            ),
            stderr="",
        )
        with patch("profiles.base.subprocess.run", return_value=proc):
            row = qwen2_kv_profile_measurement(
                "h2o",
                "edgekv-h2o",
                Request("r1", "qa", "short"),
                ProfileSpec("h2o_heavy_hitter", "h2o", "edgekv-h2o", lossy=True),
                {"pilot_model": "/models/qwen", "max_new_tokens": 1},
            )
        self.assertFalse(row.ok)
        self.assertFalse(row.measured)
        self.assertIn("H2O proof missing", row.error or "")
        self.assertEqual(str(row.extra["h2o_prune_events"]), "0")

    def test_qwen2_success_proof_fields_round_trip_through_rows(self) -> None:
        proc = Mock(
            returncode=0,
            stdout=(
                '{"ok": true, "measured": true, "output_text": "x", "latency_ms": 2.0, "ttft_ms": 1.0, '
                '"peak_memory_mib": 3.0, "resident_memory_mib": 4.0, "backend": "qwen2_kivi", '
                '"kivi_kernel_calls": 2, "kivi_quantized_layers": 1}\n'
            ),
            stderr="",
        )
        with patch("profiles.base.subprocess.run", return_value=proc):
            row = qwen2_kv_profile_measurement(
                "kivi",
                "edgekv-kivi",
                Request("r1", "qa", "prompt"),
                ProfileSpec("kivi_4bit", "kivi", "edgekv-kivi", lossy=True, metadata={"bits": 4}),
                {"pilot_model": "/models/qwen", "max_new_tokens": 1},
            )
        hydrated = ProfileMeasurement.from_row(row.to_row())
        self.assertTrue(hydrated.ok)
        self.assertTrue(hydrated.measured)
        self.assertEqual(str(hydrated.extra["kivi_kernel_calls"]), "2")
        self.assertEqual(str(hydrated.extra["kivi_quantized_layers"]), "1")

    def test_build_profile_table_no_dry_run_fails_on_kivi_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            output_path = Path(tmpdir) / "profiles.csv"
            config_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  pilot_model: /models/qwen",
                        "profiles:",
                        "  adapters:",
                        "    - kivi",
                        "  names:",
                        "    - kivi_4bit",
                        "profile_smoke:",
                        "  max_new_tokens: 1",
                        "  timeout_s: 5",
                    ]
                ),
                encoding="utf-8",
            )
            proc = Mock(
                returncode=1,
                stdout='{"ok": false, "measured": false, "error": "KIVI proof missing", "backend": "qwen2_kivi"}\n',
                stderr="",
            )
            with patch("profiles.base.subprocess.run", return_value=proc):
                code = build_profile_table(
                    argparse.Namespace(
                        config=str(config_path),
                        adapters=None,
                        output=str(output_path),
                        import_measurements="",
                        dry_run=False,
                    )
                )
            self.assertEqual(code, 2)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
