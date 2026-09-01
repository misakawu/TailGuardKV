from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import subprocess
import tempfile
import time
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import run_util.experiment as run_experiment
from calibration.conformal import ConformalGuard
from aal import AALState, AuditSample, WilsonDriftDetector
from backends.measured_replay import MeasuredReplayBackend
from run_util.core_types import Action
from run_util.core_types import BackendResult, PolicyRunRecord, ProfileMeasurement, ProfileSpec, Request
from run_util.experiment_common import annotate_measurement, config_adapters, config_policies, config_profiles, config_runtime, exact_profiles, failed_measurement_summary, limit_requests_by_split, load_config, validate_profile_measurements, with_quality, write_csv
from metrics.quality import compute_quality_loss, normalized_exact_match_loss, rouge_l_loss, select_primary_loss, token_f1_loss
from metrics import MetricCollector
from run_util.profile_summary import profile_summary_rows
from policies.base import Policy
from policies.registry import build_policies
from profiles import base as profiles_base
from profiles.base import qwen2_kv_profile_many_measurements, qwen2_kv_profile_measurement, transformers_profile_many_measurements, transformers_profile_measurement
from profiles.full import FullKVAdapter
from profiles.h2o import H2OAdapter
from profiles.kivi import KIVIAdapter
from profiles.h2o_cache import H2OCache
from profiles.kivi_cache import KIVICache
from profiles.generation_timing import generate_with_first_token_timing
from profiles import qwen2_kv_runtime
from profiles.registry import build_profile_adapters
from run_util.vllm_lru_policy import create_vllm_policy
from env_asset_prepare.prepare_pilot_assets import format_longbench_prompt
import run_util.build_profile_table as profile_table_module
from run_util.build_profile_table import build_profile_table
from run_util.cli_common import run_command
from run_util.experiment import _policy_output_for_sweep, _policy_sweep_points, _summary_rows, build_parser as build_experiment_parser, pilot_smoke_measured
from run_util.experiment_summary import summary_rows, total_policy_summary_rows
from run_util.profile_test import build_parser as build_profile_test_parser, main as run_profile_test_main
from run_util.run_policies import run_policies


def _measurement(
    request_id: str,
    profile: str,
    quality_loss: float | None,
    ttft_ms: float = 10.0,
    peak_memory_mib: float = 100.0,
    kv_cache_memory_mib: float | None = None,
    measured: bool = True,
) -> ProfileMeasurement:
    if kv_cache_memory_mib is None:
        kv_cache_memory_mib = peak_memory_mib
    return ProfileMeasurement(
        request_id=request_id,
        profile=profile,
        adapter="full",
        ok=True,
        measured=measured,
        output_text=f"{request_id}:{profile}",
        ttft_ms=ttft_ms,
        latency_ms=max(ttft_ms, 0.0) + 1.0,
        peak_memory_mib=peak_memory_mib,
        kv_cache_memory_mib=kv_cache_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=quality_loss,
        extra={"task": "qa", "length_bucket": "short", "split": "calibration"},
    )


FORMAL_PROFILES = [
    "full_gpu",
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
]

SESSION_TRACE_PROFILES = [
    *FORMAL_PROFILES,
    "kivi_4bit_residual16",
    "kivi_2bit_residual16",
    "h2o_heavy08_recent08",
    "h2o_heavy05_recent05",
]


def _write_pilot_test_config(
    path: Path,
    profile_output: Path,
    policy_output: Path,
    summary_output: Path,
    *,
    require_ttft: bool = False,
) -> None:
    profile_names = "\n".join(f"    - {profile}" for profile in FORMAL_PROFILES)
    specs = "\n".join(f"    {profile}: {{exact: {str(profile == 'full_gpu').lower()}}}" for profile in FORMAL_PROFILES)
    path.write_text(
        "\n".join(
            [
                "profiles:",
                "  adapters: [full, kivi, h2o]",
                "  names:",
                profile_names,
                "  specs:",
                specs,
                "policies:",
                "  record_rejected_unsafe: true",
                "  names: [full_lru, static_best, static_safe, tailguard, quality_oracle, utility_dynamic, uncalibrated_dynamic]",
                "pilot:",
                "  epsilons: [0.05]",
                "  deltas: [0.05]",
                "  memory_budgets_mib: [4900]",
                "outputs:",
                f"  smoke_profiles: {profile_output}",
                f"  smoke_policy: {policy_output}",
                f"  smoke_summary: {summary_output}",
                "profile_smoke:",
                f"  require_ttft: {str(require_ttft).lower()}",
            ]
        ),
        encoding="utf-8",
    )


class TailGuardCoreTest(unittest.TestCase):
    def test_root_run_launchers_are_thin_import_shims(self) -> None:
        expected = {
            "run_experiment.py": "run_util.experiment",
            "run_profile_test.py": "run_util.profile_test",
            "run_mem_test.py": "run_util.mem_test",
        }
        for filename, module in expected.items():
            with self.subTest(filename=filename):
                tree = ast.parse(Path(filename).read_text(encoding="utf-8"), filename=filename)
                definitions = [
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                self.assertEqual(definitions, [])
                self.assertTrue(
                    any(
                        isinstance(node, ast.ImportFrom)
                        and node.module == module
                        and any(alias.name == "main" for alias in node.names)
                        for node in tree.body
                    )
                )

    def test_format_longbench_prompt_includes_context_and_question(self) -> None:
        row = {
            "context": "Paper context sentence. " * 40,
            "input": "What is the result?",
        }

        prompt = format_longbench_prompt(row)

        self.assertIn("Context:\nPaper context sentence.", prompt)
        self.assertIn("\n\nQuestion:\nWhat is the result?", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))
        self.assertGreater(len(prompt), 900)

    def test_format_longbench_prompt_caps_pilot_context_without_dropping_question(self) -> None:
        row = {
            "context": "Long context sentence. " * 1000,
            "input": "What is the result?",
        }

        prompt = format_longbench_prompt(row, max_chars=1200)

        self.assertLessEqual(len(prompt), 1200)
        self.assertTrue(prompt.startswith("Context:\nLong context sentence."))
        self.assertIn("\n\nQuestion:\nWhat is the result?", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))

    def test_qwen2_runtime_dispatches_parameterized_h2o_profiles(self) -> None:
        calls = []

        def fake_h2o(payload):
            calls.append(payload["profile"])
            return {"ok": True, "measured": True, "backend": "qwen2_h2o"}

        with patch.object(qwen2_kv_runtime, "_run_h2o_profile", side_effect=fake_h2o):
            result = qwen2_kv_runtime.run_profile({"profile": "h2o_heavy10_recent10"})

        self.assertTrue(result["ok"])
        self.assertEqual(calls, ["h2o_heavy10_recent10"])

    def test_transformers_profile_many_uses_single_subprocess_for_chunk(self) -> None:
        requests = [
            Request(request_id="r1", task="summary", prompt="a"),
            Request(request_id="r2", task="summary", prompt="b"),
        ]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        with patch("profiles.base.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "ok": True,
                                "measured": True,
                                "output_text": "x",
                                "ttft_ms": 1,
                                "latency_ms": 2,
                                "peak_memory_mib": 3,
                                "resident_memory_mib": 4,
                                "stage_total_ms": 5,
                            },
                            {
                                "ok": True,
                                "measured": True,
                                "output_text": "y",
                                "ttft_ms": 1,
                                "latency_ms": 2,
                                "peak_memory_mib": 3,
                                "resident_memory_mib": 4,
                                "stage_total_ms": 5,
                            },
                        ],
                        "worker": {"mode": "batch"},
                    }
                ),
                stderr="",
            )

            rows = transformers_profile_many_measurements("full", "tailguardkv-base", requests, spec, {"max_new_tokens": 4, "pilot_model": "/tmp/model"})

        self.assertEqual(len(rows), 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([row.request_id for row in rows], ["r1", "r2"])
        self.assertEqual(rows[0].extra["worker_mode"], "batch")
        self.assertEqual(rows[0].extra["stage_total_ms"], 5)

    def test_worker_result_mapping_preserves_distinct_latency_and_ttft(self) -> None:
        request = Request("r1", "summary", "prompt")
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        row = profiles_base._measurement_from_result(
            "full",
            request,
            spec,
            {
                "ok": True,
                "measured": True,
                "output_text": "x",
                "latency_ms": 12,
                "ttft_ms": 3,
                "peak_memory_mib": 4,
                "kv_cache_memory_mib": 2,
                "resident_memory_mib": 5,
                "ttft_semantics": "first_token",
            },
            default_extra={"backend": "transformers"},
            worker_mode="batch",
        )

        self.assertEqual(row.latency_ms, 12)
        self.assertEqual(row.ttft_ms, 3)
        self.assertEqual(row.extra["ttft_semantics"], "first_token")

    def test_generation_timing_measures_first_token_and_decodes_generated_tokens(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values
                if values and isinstance(values[0], list):
                    self.shape = (len(values), len(values[0]))
                else:
                    self.shape = (1, len(values))

            def to(self, _device):
                return self

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    return self
                return self.values[index]

            def item(self):
                if self.values and isinstance(self.values[0], list):
                    return self.values[0][0]
                return self.values[0]

        class FakeLogits:
            def __init__(self, next_token):
                self.next_token = next_token

            def __getitem__(self, _index):
                return self

        class FakeOutput:
            def __init__(self, next_token, cache):
                self.logits = FakeLogits(next_token)
                self.past_key_values = cache

        class FakeTokenizer:
            eos_token_id = 99

            def decode(self, tokens, skip_special_tokens=True):
                return ",".join(str(int(token)) for token in tokens)

        class FakeModel:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                next_token = 20 + len(self.calls)
                return FakeOutput(next_token, {"cache_call": len(self.calls)})

            def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs):
                return {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                    "use_cache": use_cache,
                    "prepared": True,
                }

        class FakeCuda:
            def __init__(self):
                self.syncs = 0

            def synchronize(self, _device):
                self.syncs += 1

        class FakeTorch:
            def __init__(self):
                self.cuda = FakeCuda()

            @staticmethod
            def argmax(logits, dim=-1, keepdim=True):
                return FakeTensor([[logits.next_token]])

        model = FakeModel()
        torch = FakeTorch()
        request_start = time.perf_counter()

        result = generate_with_first_token_timing(
            model,
            FakeTokenizer(),
            torch,
            {"input_ids": FakeTensor([[11, 12, 13]]), "attention_mask": FakeTensor([[1, 1, 1]])},
            request_start=request_start,
            device="cuda:0",
            max_new_tokens=3,
            has_cuda=True,
        )

        self.assertEqual(result["output_text"], "21,22,23")
        self.assertIsNotNone(result["ttft_ms"])
        self.assertLessEqual(result["ttft_ms"], result["stage_generate_ms"])
        self.assertEqual(result["stage_first_token_ms"], result["ttft_ms"])
        self.assertGreaterEqual(result["stage_prefill_ms"], 0.0)
        self.assertEqual(len(model.calls), 3)
        self.assertNotIn("past_key_values", model.calls[0])
        self.assertTrue(model.calls[1]["prepared"])
        self.assertEqual(result["past_key_values"], {"cache_call": 3})
        self.assertGreaterEqual(torch.cuda.syncs, 1)

    def test_transformers_runtime_uses_first_token_semantics(self) -> None:
        source = Path("profiles/transformers_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('"ttft_ms": total_ms', source)
        self.assertIn('"ttft_semantics": "first_token"', source)

    def test_mem_test_parser_defaults_cover_10_to_100_mib_budget_sweep(self) -> None:
        from run_util.mem_test import build_parser

        args = build_parser().parse_args([])

        self.assertEqual(args.budget_start_mib, 10.0)
        self.assertEqual(args.budget_stop_mib, 100.0)
        self.assertEqual(args.budget_step_mib, 10.0)

    def test_mem_test_budget_series_uses_10_mib_steps(self) -> None:
        from run_util.mem_test_config import build_budget_series

        self.assertEqual(
            build_budget_series(10, 100, 10),
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        )
        self.assertEqual(build_budget_series(100, 500, 100), [100.0, 200.0, 300.0, 400.0, 500.0])
        self.assertEqual(build_budget_series(100, 350, 100), [100.0, 200.0, 300.0])

    def test_mem_test_default_run_dir_uses_month_day_hour_prefix(self) -> None:
        from run_util import mem_test

        fixed_time = SimpleNamespace(strftime=lambda fmt: "08-04-15" if fmt == "%m-%d-%H" else "20260804")
        with patch("run_util.mem_test.datetime", SimpleNamespace(now=lambda: fixed_time)):
            run_dir = mem_test._resolve_run_dir(None)

        self.assertEqual(run_dir, Path("out/08-04-15_mem_test"))

    def test_mem_test_generated_config_removes_tailguard_and_uses_relative_outputs(self) -> None:
        from run_util.mem_test_config import build_mem_test_config

        base = load_config(Path("configs/pilot.yaml"))
        config = build_mem_test_config(base, max_requests=80, budgets_mib=[100.0, 200.0], include_tailguard=False)

        self.assertEqual(config["data"]["max_requests"], 80)
        self.assertEqual(config["pilot"]["memory_budgets_mib"], [100.0, 200.0])
        self.assertNotIn("tailguard", config["policies"]["names"])
        self.assertEqual(
            config["policies"]["names"],
            ["full_lru", "static_best", "static_safe", "quality_oracle", "utility_dynamic", "uncalibrated_dynamic"],
        )
        self.assertEqual(config["outputs"]["smoke_profiles"], "out/profile_tables/run_mem_test_profiles.csv")
        self.assertEqual(config["outputs"]["smoke_policy"], "out/policy_tables/run_mem_test_policy.csv")
        self.assertEqual(config["outputs"]["smoke_summary"], "out/policy_tables/run_mem_test_summary.csv")

    def test_mem_test_analysis_finds_budget_with_kv_and_ttft_gain(self) -> None:
        from run_util.mem_test_analysis import analyze_mem_test_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,mean_ttft_ms,p95_ttft_ms,mean_kv_cache_memory_mib,p95_kv_cache_memory_mib,action_distribution,violation_rate,candidate_safe_count",
                        'full_lru,100,0.05,0.05,100,200,80,90,"{""full_gpu"": 2}",0,',
                        'static_best,100,0.05,0.05,90,210,30,35,"{""h2o_heavy15_recent15"": 2}",0.05,',
                    ]
                ),
                encoding="utf-8",
            )

            analysis = analyze_mem_test_summary(path)

        self.assertTrue(analysis["found_passing_budget"])
        self.assertEqual(analysis["passing_points"][0]["policy"], "static_best")
        self.assertEqual(analysis["passing_points"][0]["memory_budget_mib"], 100.0)
        self.assertEqual(analysis["passing_points"][0]["ttft_win_metric"], "mean_ttft_ms")

    def test_mem_test_analysis_reports_no_budget_when_ttft_not_better(self) -> None:
        from run_util.mem_test_analysis import analyze_mem_test_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,mean_ttft_ms,p95_ttft_ms,mean_kv_cache_memory_mib,p95_kv_cache_memory_mib,action_distribution,violation_rate,candidate_safe_count",
                        'full_lru,100,0.05,0.05,100,200,80,90,"{""full_gpu"": 2}",0,',
                        'static_best,100,0.05,0.05,120,220,30,35,"{""h2o_heavy15_recent15"": 2}",0.05,',
                    ]
                ),
                encoding="utf-8",
            )

            analysis = analyze_mem_test_summary(path)

        self.assertFalse(analysis["found_passing_budget"])
        self.assertEqual(analysis["near_misses"][0]["policy"], "static_best")
        self.assertEqual(analysis["near_misses"][0]["kv_drop_mean"], 0.625)

    def test_mem_test_analysis_records_ttft_gain_but_kv_miss_as_near_miss(self) -> None:
        from run_util.mem_test_analysis import analyze_mem_test_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,mean_ttft_ms,p95_ttft_ms,mean_kv_cache_memory_mib,p95_kv_cache_memory_mib,action_distribution,violation_rate,candidate_safe_count",
                        'full_lru,100,0.05,0.05,100,200,80,90,"{""full_gpu"": 2}",0,',
                        'static_best,100,0.05,0.05,90,190,45,50,"{""h2o_heavy15_recent15"": 2}",0.05,',
                    ]
                ),
                encoding="utf-8",
            )

            analysis = analyze_mem_test_summary(path)

        self.assertFalse(analysis["found_passing_budget"])
        self.assertEqual(analysis["near_misses"][0]["policy"], "static_best")
        self.assertEqual(analysis["near_misses"][0]["kv_drop_mean"], 0.4375)
        self.assertNotIn("ttft_win_metric", analysis["near_misses"][0])

    def test_mem_test_analysis_ignores_non_kivi_h2o_lossy_actions(self) -> None:
        from run_util.mem_test_analysis import analyze_mem_test_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,mean_ttft_ms,p95_ttft_ms,mean_kv_cache_memory_mib,p95_kv_cache_memory_mib,action_distribution,violation_rate,candidate_safe_count",
                        'full_lru,100,0.05,0.05,100,200,80,90,"{""full_gpu"": 2}",0,',
                        'static_best,100,0.05,0.05,90,190,30,35,"{""adaptive_budget"": 2, ""full_gpu"": 1}",0.05,',
                    ]
                ),
                encoding="utf-8",
            )

            analysis = analyze_mem_test_summary(path)

        self.assertFalse(analysis["found_passing_budget"])
        self.assertEqual(analysis["passing_points"], [])
        self.assertEqual(analysis["near_misses"], [])

    def test_run_mem_test_orchestrates_baseline_only_sweep_without_nested_outputs(self) -> None:
        from run_util import mem_test

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "mem_run"
            config_path = root / "pilot.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  pilot_model: fake",
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  record_rejected_unsafe: true",
                        "  names: [full_lru, tailguard]",
                        "pilot:",
                        "  epsilons: [0.05]",
                        "  deltas: [0.05]",
                        "  memory_budgets_mib: [4900]",
                        "data:",
                        "  requests: fake.jsonl",
                        "  calibration_fraction: 0.5",
                        "  max_requests: 200",
                        "profile_smoke:",
                        "  repeat: 1",
                        "outputs:",
                        "  smoke_profiles: out/profile_tables/pilot.csv",
                        "  smoke_policy: out/policy_tables/policy.csv",
                        "  smoke_summary: out/policy_tables/summary.csv",
                    ]
                ),
                encoding="utf-8",
            )
            calls = {"policy": []}

            def fake_build(args: argparse.Namespace) -> int:
                calls["profile"] = args
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1}}}))
                return 0

            def fake_policy(args: argparse.Namespace) -> int:
                calls["policy"].append(args)
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 1,
                            "epsilon": args.epsilon,
                            "delta": args.delta,
                            "memory_budget_mib": args.memory_budget_mib,
                            "summary": {
                                "full_lru": {
                                    "mean_ttft_ms": 100.0,
                                    "p95_ttft_ms": 200.0,
                                    "mean_kv_cache_memory_mib": 80.0,
                                    "p95_kv_cache_memory_mib": 90.0,
                                    "violation_rate": 0.0,
                                    "action_distribution": {"full_gpu": 1},
                                }
                            },
                        }
                    )
                )
                return 0

            with (
                patch("run_util.mem_test.build_profile_table", side_effect=fake_build),
                patch("run_util.mem_test.run_policies", side_effect=fake_policy),
                patch("run_util.mem_test.plot_summary", return_value=[]),
            ):
                args = mem_test.build_parser().parse_args(
                    [
                        "--base-config",
                        str(config_path),
                        "--run-dir",
                        str(run_dir),
                    ]
                )
                code = mem_test.run(args)

            self.assertEqual(code, 0)
            expected_budgets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            generated_config = load_config(run_dir / "configs" / "run_mem_test.generated.yaml")
            self.assertEqual(generated_config["pilot"]["memory_budgets_mib"], expected_budgets)
            self.assertFalse(calls["profile"].formal_run)
            self.assertEqual(calls["profile"].output, str(run_dir / "profile_tables/run_mem_test_profiles.csv"))
            self.assertEqual(
                [call.memory_budget_mib for call in calls["policy"]],
                expected_budgets,
            )
            self.assertTrue(all(call.allow_dry_run_replay is False for call in calls["policy"]))
            self.assertTrue((run_dir / "policy_tables/run_mem_test_total_summary.csv").exists())
            self.assertTrue((run_dir / "run_mem_test_analysis.md").exists())
            with (run_dir / "policy_tables/run_mem_test_summary.csv").open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["section"], "experiment")
            self.assertEqual(summary_rows[0]["name"], "run_mem_test")
            self.assertFalse((run_dir / "mem_run").exists())

    def test_qwen2_generate_decode_uses_first_token_semantics(self) -> None:
        source = Path("profiles/qwen2_runtime_common.py").read_text(encoding="utf-8")
        self.assertNotIn('"ttft_ms": total_ms', source)
        self.assertIn('"ttft_semantics": "first_token"', source)

    def test_transformers_profile_many_uses_pilot_model_only(self) -> None:
        requests = [Request(request_id="r1", task="summary", prompt="a")]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        def fake_run(command, **kwargs):
            payload = json.loads(kwargs["env"]["TRANSFORMERS_PROFILE_PAYLOAD"])
            self.assertEqual(payload["requests"][0]["model_name"], "/tmp/pilot")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "ok": True,
                                "measured": True,
                                "output_text": "x",
                                "ttft_ms": 1,
                                "latency_ms": 2,
                                "peak_memory_mib": 3,
                                "resident_memory_mib": 4,
                            }
                        ],
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            rows = transformers_profile_many_measurements(
                "full",
                "tailguardkv-base",
                requests,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/pilot"},
            )

        self.assertTrue(rows[0].ok)

    def test_transformers_profile_measurement_uses_pilot_model_only(self) -> None:
        request = Request(request_id="r1", task="summary", prompt="a")
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        def fake_run(command, **kwargs):
            code = command[-1]
            self.assertIn("'model_name': '/tmp/pilot'", code)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "output_text": "x",
                        "ttft_ms": 1,
                        "latency_ms": 2,
                        "peak_memory_mib": 3,
                        "resident_memory_mib": 4,
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            row = transformers_profile_measurement(
                "full",
                "tailguardkv-base",
                request,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/pilot"},
            )

        self.assertTrue(row.ok)

    def test_qwen2_profile_measurement_falls_back_to_pilot_model(self) -> None:
        request = Request(request_id="r1", task="summary", prompt="a")
        spec = ProfileSpec("kivi_4bit_residual32", "kivi", "tailguardkv-base", lossy=True)

        def fake_run(command, **kwargs):
            payload = json.loads(kwargs["env"]["QWEN2_KV_PAYLOAD"])
            self.assertEqual(payload["model_name"], "/tmp/pilot")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "measured": True,
                        "output_text": "x",
                        "ttft_ms": 1,
                        "latency_ms": 2,
                        "peak_memory_mib": 3,
                        "resident_memory_mib": 4,
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            row = qwen2_kv_profile_measurement(
                "kivi",
                "tailguardkv-base",
                request,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/pilot"},
            )

        self.assertTrue(row.ok)

    def test_qwen2_profile_measurement_appends_pythonpath(self) -> None:
        request = Request(request_id="r1", task="summary", prompt="a")
        spec = ProfileSpec("h2o_heavy10_recent10", "h2o", "edgekv-h2o", lossy=True)

        def fake_run(command, **kwargs):
            pythonpath = kwargs["env"]["PYTHONPATH"].split(":")
            self.assertIn("/DATACENTER3/zhenxiang.wang/work/TailGuardKV/third_party/H2O/h2o_hf", pythonpath)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "measured": True,
                        "output_text": "x",
                        "ttft_ms": 1,
                        "latency_ms": 2,
                        "peak_memory_mib": 3,
                        "resident_memory_mib": 4,
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            row = qwen2_kv_profile_measurement(
                "h2o",
                "edgekv-h2o",
                request,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/pilot"},
                pythonpath=("third_party/H2O/h2o_hf",),
            )

        self.assertTrue(row.ok)

    def test_batch_runtime_maps_worker_failure_to_every_request(self) -> None:
        requests = [
            Request(request_id="r1", task="summary", prompt="a"),
            Request(request_id="r2", task="summary", prompt="b"),
        ]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        with patch("profiles.base.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="CUDA out of memory")

            rows = transformers_profile_many_measurements("full", "tailguardkv-base", requests, spec, {"max_new_tokens": 4, "pilot_model": "/tmp/model"})

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row.ok and not row.measured for row in rows))
        self.assertTrue(all("CUDA out of memory" in (row.error or "") for row in rows))

    def test_batch_timeout_scales_with_request_count(self) -> None:
        self.assertEqual(profiles_base._batch_timeout_s({"timeout_s": 180}, request_count=10), 1800)

    def test_batch_timeout_keeps_single_request_budget(self) -> None:
        self.assertEqual(profiles_base._batch_timeout_s({"timeout_s": 180}, request_count=1), 180)

    def test_transformers_profile_many_uses_scaled_batch_timeout(self) -> None:
        requests = [
            Request(request_id="r1", task="summary", prompt="a"),
            Request(request_id="r2", task="summary", prompt="b"),
            Request(request_id="r3", task="summary", prompt="c"),
            Request(request_id="r4", task="summary", prompt="d"),
            Request(request_id="r5", task="summary", prompt="e"),
            Request(request_id="r6", task="summary", prompt="f"),
            Request(request_id="r7", task="summary", prompt="g"),
            Request(request_id="r8", task="summary", prompt="h"),
            Request(request_id="r9", task="summary", prompt="i"),
            Request(request_id="r10", task="summary", prompt="j"),
        ]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        def fake_run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 1800)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "ok": True,
                                "measured": True,
                                "output_text": request.request_id,
                                "ttft_ms": 1,
                                "latency_ms": 2,
                                "peak_memory_mib": 3,
                                "resident_memory_mib": 4,
                            }
                            for request in requests
                        ],
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            rows = transformers_profile_many_measurements(
                "full",
                "tailguardkv-base",
                requests,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/model", "timeout_s": 180},
            )

        self.assertEqual(len(rows), 10)

    def test_transformers_profile_many_forwards_session_runtime_and_budget(self) -> None:
        requests = [
            Request(request_id="r1", task="summary", prompt="hello", session_id="s1", turn_index=0),
        ]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)
        runtime_state = {"sessions": {"s1": {"current_profile": "full_gpu"}}}

        def fake_run(command, **kwargs):
            payload = json.loads(kwargs["env"]["TRANSFORMERS_PROFILE_PAYLOAD"])
            self.assertEqual(payload["session_runtime_state"], runtime_state)
            self.assertEqual(payload["memory_budget_mib"], 64.0)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "ok": True,
                                "measured": True,
                                "output_text": "x",
                                "ttft_ms": 1,
                                "latency_ms": 2,
                                "peak_memory_mib": 3,
                                "resident_memory_mib": 4,
                                "kv_incremental_mib": 3,
                                "kv_cumulative_mib": 3,
                                "resident_kv_mib_before": 0,
                                "resident_kv_mib_after": 3,
                                "budget_hit": False,
                            }
                        ],
                        "session_runtime_state": runtime_state,
                    }
                ),
                stderr="",
            )

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            rows = transformers_profile_many_measurements(
                "full",
                "tailguardkv-base",
                requests,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/model"},
                session_runtime=runtime_state,
                memory_budget_mib=64.0,
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].ok)

    def test_batch_runtime_timeout_returns_structured_failure_rows(self) -> None:
        requests = [
            Request(request_id="r1", task="summary", prompt="a"),
            Request(request_id="r2", task="summary", prompt="b"),
        ]
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        def fake_run(command, **kwargs):
            raise subprocess.TimeoutExpired(cmd=command, timeout=360)

        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            rows = transformers_profile_many_measurements(
                "full",
                "tailguardkv-base",
                requests,
                spec,
                {"max_new_tokens": 4, "pilot_model": "/tmp/model", "timeout_s": 180},
                timeout_s=360,
            )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row.ok and not row.measured for row in rows))
        self.assertTrue(all("profiles.transformers_runtime" in (row.error or "") for row in rows))
        self.assertTrue(all("360" in (row.error or "") for row in rows))
        self.assertTrue(all(row.extra["error_type"] == "timeout" for row in rows))
        self.assertTrue(all(row.extra["failure_stage"] == "worker_startup" for row in rows))

    def test_failed_measurement_summary_keeps_timeout_failures(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="full_gpu",
            adapter="full",
            ok=False,
            measured=False,
            error="profiles.transformers_runtime 启动超时: TimeoutExpired after 360s",
            extra={"backend": "transformers", "error_type": "timeout", "failure_stage": "worker_startup"},
        )

        summary = failed_measurement_summary([row])

        self.assertEqual(len(summary), 1)
        self.assertIn("TimeoutExpired", str(summary[0]["error"]))

    def test_full_adapter_measured_profile_many_uses_batch_runtime(self) -> None:
        adapter = FullKVAdapter({"max_new_tokens": 4})
        requests = [Request(request_id="r1", task="summary", prompt="a")]

        with patch("profiles.full.qwen2_exact_profile_many_measurements") as many:
            many.return_value = [_measurement("r1", "full_gpu", 0.0)]
            rows = adapter.profile_many(requests, "full_gpu", dry_run=False)

        many.assert_called_once()
        self.assertEqual(rows[0].request_id, "r1")

    def test_kivi_adapter_measured_profile_many_uses_batch_runtime(self) -> None:
        adapter = KIVIAdapter({"max_new_tokens": 4})
        requests = [Request(request_id="r1", task="summary", prompt="a")]
        session_runtime = {"state": "sentinel"}

        with patch("profiles.kivi.qwen2_kv_profile_many_measurements") as many:
            many.return_value = [_measurement("r1", "kivi_4bit_residual32", None)]
            rows = adapter.profile_many(
                requests,
                "kivi_4bit_residual32",
                dry_run=False,
                session_runtime=session_runtime,
                memory_budget_mib=64.0,
            )

        many.assert_called_once()
        self.assertIs(many.call_args.kwargs["session_runtime"], session_runtime)
        self.assertEqual(many.call_args.kwargs["memory_budget_mib"], 64.0)
        self.assertEqual(rows[0].profile, "kivi_4bit_residual32")

    def test_h2o_adapter_measured_profile_many_uses_batch_runtime_and_pythonpath(self) -> None:
        adapter = H2OAdapter({"max_new_tokens": 4})
        requests = [
            Request(request_id="r1", task="summary", prompt="a"),
            Request(request_id="r2", task="summary", prompt="b"),
        ]
        session_runtime = {"state": "sentinel"}

        with patch("profiles.h2o.qwen2_kv_profile_many_measurements") as many:
            many.return_value = [
                _measurement("r1", "h2o_heavy10_recent10", None),
                _measurement("r2", "h2o_heavy10_recent10", None),
            ]
            rows = adapter.profile_many(
                requests,
                "h2o_heavy10_recent10",
                dry_run=False,
                session_runtime=session_runtime,
                memory_budget_mib=64.0,
            )

        many.assert_called_once()
        self.assertEqual(many.call_args.kwargs["pythonpath"], adapter.pythonpath)
        self.assertIs(many.call_args.kwargs["session_runtime"], session_runtime)
        self.assertEqual(many.call_args.kwargs["memory_budget_mib"], 64.0)
        self.assertEqual([row.request_id for row in rows], ["r1", "r2"])
        self.assertTrue(all(row.profile == "h2o_heavy10_recent10" for row in rows))

    def test_vllm_adapter_is_registered(self) -> None:
        adapters = build_profile_adapters(["vllm"], {"max_new_tokens": 1})
        self.assertEqual([adapter.name for adapter in adapters], ["vllm"])
        self.assertEqual(adapters[0].profile_names(), ("engine_full_lru",))

    def test_vllm_worker_failure_does_not_call_transformers_helpers(self) -> None:
        from profiles.vllm import VLLMAdapter

        adapter = VLLMAdapter({"pilot_model": "/tmp/model", "max_new_tokens": 1})
        with (
            patch("profiles.vllm.subprocess.run", side_effect=RuntimeError("vllm import failed")),
            patch("profiles.vllm.dry_profile_measurement") as dry,
            patch("profiles.base.transformers_profile_many_measurements", side_effect=AssertionError("no transformers fallback")),
        ):
            rows = adapter.profile_many([Request("r1", "qa", "prompt")], "engine_full_lru", dry_run=False)

        self.assertFalse(rows[0].ok)
        self.assertFalse(rows[0].measured)
        self.assertEqual(rows[0].extra["backend"], "vllm")
        dry.assert_not_called()

    def test_vllm_successful_worker_row_marks_backend_and_measured(self) -> None:
        from profiles.vllm import VLLMAdapter

        adapter = VLLMAdapter({"pilot_model": "/tmp/model", "max_new_tokens": 1})

        def fake_run(command, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "request_id": "r1",
                                "ok": True,
                                "output_text": "x",
                                "latency_ms": 12,
                                "ttft_ms": None,
                                "peak_memory_mib": 30,
                                "kv_cache_memory_mib": 4,
                                "resident_memory_mib": 30,
                                "ttft_semantics": "unavailable",
                            }
                        ],
                    }
                ),
                stderr="",
            )

        with patch("profiles.vllm.subprocess.run", side_effect=fake_run):
            rows = adapter.profile_many([Request("r1", "qa", "prompt")], "engine_full_lru", dry_run=False)

        self.assertTrue(rows[0].measured)
        self.assertEqual(rows[0].extra["backend"], "vllm")
        self.assertEqual(rows[0].kv_cache_memory_mib, 4)

    def test_vllm_worker_uses_last_token_time_as_ttft_for_one_token_generation(self) -> None:
        from profiles.vllm_worker import _ttft_from_metrics

        metrics = SimpleNamespace(arrival_time=10.0, first_token_time=None, last_token_time=10.25)

        ttft_ms, semantics = _ttft_from_metrics(metrics, max_new_tokens=1)

        self.assertEqual(ttft_ms, 250.0)
        self.assertEqual(semantics, "first_token")

    def test_vllm_worker_uses_request_end_as_ttft_for_one_token_without_metrics(self) -> None:
        from profiles.vllm_worker import _ttft_from_metrics

        ttft_ms, semantics = _ttft_from_metrics(None, max_new_tokens=1, request_start=10.0, request_end=10.5)

        self.assertEqual(ttft_ms, 500.0)
        self.assertEqual(semantics, "first_token")

    def test_qwen2_runtime_batch_dispatches_each_profile_in_order(self) -> None:
        calls = []

        def fake_run(payload):
            calls.append(payload["profile"])
            return {
                "ok": True,
                "measured": True,
                "output_text": payload["profile"],
                "ttft_ms": 1,
                "latency_ms": 1,
                "peak_memory_mib": 1,
                "resident_memory_mib": 1,
            }

        with patch.object(qwen2_kv_runtime, "run_profile", side_effect=fake_run):
            result = qwen2_kv_runtime.run_profile_batch(
                {
                    "requests": [
                        {"profile": "kivi_4bit_residual32"},
                        {"profile": "h2o_heavy10_recent10"},
                    ]
                }
            )

        self.assertEqual(calls, ["kivi_4bit_residual32", "h2o_heavy10_recent10"])
        self.assertEqual(
            [item["output_text"] for item in result["results"]],
            ["kivi_4bit_residual32", "h2o_heavy10_recent10"],
        )

    def test_qwen2_batch_runtime_preserves_stage_timings_in_results(self) -> None:
        payload = {
            "requests": [
                {
                    "profile": "kivi_4bit_residual32",
                    "prompt": "p",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 2,
                }
            ]
        }

        with patch.object(
            qwen2_kv_runtime,
            "_run_kivi_profile",
            return_value={
                "ok": True,
                "measured": True,
                "output_text": "x",
                "ttft_ms": 10,
                "latency_ms": 12,
                "peak_memory_mib": 100,
                "resident_memory_mib": 200,
                "stage_model_load_ms": 300,
                "stage_generate_ms": 40,
                "stage_total_ms": 352,
            },
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertEqual(result["results"][0]["stage_model_load_ms"], 300)
        self.assertEqual(result["results"][0]["stage_total_ms"], 352)

    def test_build_profile_table_writes_stage_timing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profiles.csv"
            row = ProfileMeasurement(
                request_id="r1",
                profile="full_gpu",
                adapter="full",
                ok=True,
                measured=True,
                output_text="x",
                ttft_ms=1.0,
                peak_memory_mib=2.0,
                resident_memory_mib=3.0,
                quality_loss=0.0,
                extra={
                    "task": "summary",
                    "length_bucket": "short",
                    "split": "calibration",
                    "stage_prefill_ms": 2.0,
                    "stage_first_token_ms": 3.0,
                    "stage_total_ms": 9.0,
                },
            )

            profile_table_module._append_profile_rows(output, [row.to_row()])
            text = output.read_text(encoding="utf-8")

        self.assertIn("extra_stage_total_ms", text)
        self.assertIn("extra_stage_prefill_ms", text)
        self.assertIn("extra_stage_first_token_ms", text)

    def test_batch_measurements_keep_successful_rows_when_worker_returns_non_zero_for_other_items(self) -> None:
        request = Request("r1", "summary", "prompt", metadata={"split": "calibration"})
        spec = ProfileSpec("kivi_4bit_residual32", "kivi", "edgekv-kivi", lossy=True)
        proc = subprocess.CompletedProcess(
            args=["conda", "run"],
            returncode=1,
            stdout='{"ok": false}',
            stderr="worker saw some item failures",
        )
        result = {
            "ok": False,
            "worker": {"mode": "batch"},
            "results": [
                {
                    "ok": True,
                    "measured": True,
                    "output_text": "continued",
                    "ttft_ms": 10.0,
                    "latency_ms": 12.0,
                    "peak_memory_mib": 14.0,
                    "kv_cache_memory_mib": 14.0,
                    "resident_memory_mib": 14.0,
                }
            ],
        }

        [row] = profiles_base._measurements_from_batch_result(
            "kivi",
            [request],
            spec,
            proc,
            result,
            default_extra={"backend": "qwen2_kivi"},
        )

        self.assertTrue(row.ok)
        self.assertTrue(row.measured)
        self.assertEqual(row.output_text, "continued")
        self.assertNotIn("returncode", row.extra)

    def test_measurement_from_result_preserves_kivi_quantization_flags(self) -> None:
        request = Request("r1", "summary", "prompt", metadata={"split": "calibration"})
        spec = ProfileSpec("kivi_4bit_residual64", "kivi", "edgekv-kivi", lossy=True)

        row = profiles_base._measurement_from_result(
            "kivi",
            request,
            spec,
            {
                "ok": True,
                "measured": True,
                "output_text": "ok",
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 2.0,
                "resident_memory_mib": 3.0,
                "kivi_quantization_triggered": False,
                "kivi_effective_mode": "unquantized_short_request",
            },
            default_extra={"env": "edgekv-kivi"},
            worker_mode="batch",
        )

        self.assertEqual(row.extra["kivi_quantization_triggered"], False)
        self.assertEqual(row.extra["kivi_effective_mode"], "unquantized_short_request")

    def test_measurement_from_result_preserves_binding_diagnostics(self) -> None:
        request = Request("r1", "summary", "prompt")
        spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

        row = profiles_base._measurement_from_result(
            "full",
            request,
            spec,
            {
                "ok": True,
                "measured": True,
                "output_text": "ok",
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 2.0,
                "resident_memory_mib": 3.0,
                "worker_cuda_visible_devices": "0,1",
                "runtime_cuda_visible_devices": "0,1",
                "runtime_visible_device_count": 2,
                "runtime_device_strategy": "balanced_two_gpu",
            },
            default_extra={"env": "tailguardkv-base"},
            worker_mode="persistent",
        )

        self.assertEqual(row.extra["worker_cuda_visible_devices"], "0,1")
        self.assertEqual(row.extra["runtime_cuda_visible_devices"], "0,1")
        self.assertEqual(row.extra["runtime_visible_device_count"], 2)
        self.assertEqual(row.extra["runtime_device_strategy"], "balanced_two_gpu")

    def test_append_profile_rows_keeps_diagnostic_and_kivi_columns(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="kivi_4bit_residual64",
            adapter="kivi",
            ok=True,
            measured=True,
            output_text="ok",
            latency_ms=1.0,
            ttft_ms=1.0,
            peak_memory_mib=2.0,
            resident_memory_mib=3.0,
            quality_score=1.0,
            quality_loss=0.0,
            extra={
                "task": "summary",
                "length_bucket": "short",
                "split": "calibration",
                "kivi_quantization_triggered": False,
                "kivi_effective_mode": "unquantized_short_request",
                "worker_cuda_visible_devices": "0,1",
                "worker_device_strategy": "balanced_two_gpu",
                "runtime_cuda_visible_devices": "0,1",
                "runtime_device_strategy": "balanced_two_gpu",
                "runtime_visible_device_count": 2,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.csv"
            profile_table_module._append_profile_rows(path, [row.to_row()])
            rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))

        self.assertEqual(rows[0]["extra_kivi_quantization_triggered"], "False")
        self.assertEqual(rows[0]["extra_kivi_effective_mode"], "unquantized_short_request")
        self.assertEqual(rows[0]["extra_worker_cuda_visible_devices"], "0,1")
        self.assertEqual(rows[0]["extra_worker_device_strategy"], "balanced_two_gpu")
        self.assertEqual(rows[0]["extra_runtime_cuda_visible_devices"], "0,1")
        self.assertEqual(rows[0]["extra_runtime_device_strategy"], "balanced_two_gpu")
        self.assertEqual(rows[0]["extra_runtime_visible_device_count"], "2")

    def test_qwen2_runtime_profiles_use_first_token_helper_instead_of_generate(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values
                if values and isinstance(values[0], list):
                    self.shape = (len(values), len(values[0]))
                else:
                    self.shape = (1, len(values))

            def to(self, _device):
                return self

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    return self
                return self.values[index]

            def item(self):
                if self.values and isinstance(self.values[0], list):
                    return self.values[0][0]
                return self.values[0]

        class FakeLogits:
            def __init__(self, next_token):
                self.next_token = next_token

            def __getitem__(self, _index):
                return self

        class FakeOutput:
            def __init__(self, next_token, cache):
                self.logits = FakeLogits(next_token)
                self.past_key_values = cache

        class FakeTokenizer:
            eos_token_id = 999

            def __call__(self, prompt, return_tensors="pt"):
                return {
                    "input_ids": FakeTensor([11, 12, 13, 14, 15, 16]),
                    "attention_mask": FakeTensor([1, 1, 1, 1, 1, 1]),
                }

            def decode(self, tokens, skip_special_tokens=True):
                return "generated output"

        class FakeModel:
            def __init__(self):
                self.forward_calls = []
                self.h2o_tracker = None
                self.config = SimpleNamespace(
                    num_hidden_layers=3,
                    model_type="qwen2",
                    num_attention_heads=2,
                    num_key_value_heads=1,
                )
                self.model = SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                        )
                    ]
                )

            def __call__(self, **kwargs):
                self.forward_calls.append(kwargs)
                if self.h2o_tracker is not None:
                    self.h2o_tracker["h2o_prune_events"] = 1
                    self.h2o_tracker["h2o_kept_tokens"] = self.h2o_tracker["h2o_cache_budget"]
                return FakeOutput(99 + len(self.forward_calls), kwargs.get("past_key_values"))

            def generate(self, **kwargs):
                raise AssertionError("one-shot generate path should not run")

            def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs):
                return {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                    "use_cache": use_cache,
                    "prepared": True,
                }

        class FakeCuda:
            @staticmethod
            def reset_peak_memory_stats(_device):
                return None

            @staticmethod
            def synchronize(_device):
                return None

            @staticmethod
            def max_memory_allocated(_device):
                return 4 * 1024 * 1024

            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda()
            float16 = "float16"

            @staticmethod
            def argmax(logits, dim=-1, keepdim=True):
                return FakeTensor([[logits.next_token]])

            @staticmethod
            def inference_mode():
                class _Context:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return _Context()

        runtime = {"model": None}

        def fake_load_model(payload, torch, auto_model, auto_tokenizer):
            runtime["model"] = FakeModel()
            return runtime["model"], FakeTokenizer(), "cuda:0"

        def fake_install_attention(model, wrapper_cls, tracker, bits, payload, modules):
            if str(payload["profile"]).startswith("kivi_"):
                tracker["kivi_quantized_layers"] = 1
                tracker["kivi_kernel_calls"] = 1
            else:
                model.h2o_tracker = tracker

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(qwen2_kv_runtime, "_load_qwen2_model", side_effect=fake_load_model),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention", side_effect=fake_install_attention),
            patch.object(qwen2_kv_runtime, "_greedy_decode", side_effect=AssertionError("manual decode path should not run")),
        ):
            kivi = qwen2_kv_runtime._run_kivi_profile(
                {
                    "profile": "kivi_4bit_residual32",
                    "prompt": "prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 2,
                    "kivi_residual_length": 32,
                }
            )
            kivi_model = runtime["model"]
            h2o = qwen2_kv_runtime._run_h2o_profile(
                {
                    "profile": "h2o_heavy10_recent10",
                    "prompt": "prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 2,
                    "h2o_heavy_ratio": 0.1,
                    "h2o_recent_ratio": 0.1,
                }
            )
            h2o_model = runtime["model"]

        self.assertTrue(kivi["ok"])
        self.assertTrue(h2o["ok"])
        self.assertEqual(len(kivi_model.forward_calls), 2)
        self.assertEqual(len(h2o_model.forward_calls), 2)
        self.assertIn("past_key_values", kivi_model.forward_calls[0])
        self.assertNotIsInstance(kivi_model.forward_calls[0]["past_key_values"], tuple)
        self.assertIn("past_key_values", h2o_model.forward_calls[0])
        self.assertIsInstance(h2o_model.forward_calls[0]["past_key_values"], H2OCache)
        self.assertEqual(kivi["ttft_semantics"], "first_token")
        self.assertEqual(h2o["ttft_semantics"], "first_token")

    def test_generate_decode_accepts_explicit_past_key_values(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def to(self, _device):
                return self

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    return self
                return self.values[index]

            def item(self):
                if self.values and isinstance(self.values[0], list):
                    return self.values[0][0]
                return self.values[0]

        class FakeLogits:
            def __init__(self, next_token):
                self.next_token = next_token

            def __getitem__(self, _index):
                return self

        class FakeOutput:
            def __init__(self, next_token, cache):
                self.logits = FakeLogits(next_token)
                self.past_key_values = cache

        class FakeTokenizer:
            eos_token_id = 999

            def __call__(self, prompt, return_tensors="pt"):
                return {
                    "input_ids": FakeTensor([11, 12, 13]),
                    "attention_mask": FakeTensor([1, 1, 1]),
                }

            def decode(self, tokens, skip_special_tokens=True):
                return "generated output"

        class FakeModel:
            def __init__(self):
                self.forward_kwargs = None

            def __call__(self, **kwargs):
                self.forward_kwargs = kwargs
                return FakeOutput(99, kwargs.get("past_key_values"))

        class FakeCuda:
            @staticmethod
            def reset_peak_memory_stats(_device):
                return None

            @staticmethod
            def synchronize(_device):
                return None

            @staticmethod
            def max_memory_allocated(_device):
                return 4 * 1024 * 1024

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def argmax(logits, dim=-1, keepdim=True):
                return FakeTensor([[logits.next_token]])

            @staticmethod
            def inference_mode():
                class _Context:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return _Context()

        model = FakeModel()
        cache = KIVICache(2, residual_length=32, group_size=32, k_bits=4, v_bits=4)

        result = qwen2_kv_runtime._generate_decode(
            model,
            FakeTokenizer(),
            "cuda:0",
            {"prompt": "prompt", "max_new_tokens": 1},
            FakeTorch(),
            past_key_values=cache,
        )

        self.assertTrue(result["ok"])
        self.assertIs(model.forward_kwargs["past_key_values"], cache)
        self.assertEqual(result["ttft_semantics"], "first_token")

    def test_run_kivi_profile_seeds_generate_with_kivi_cache(self) -> None:
        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                class _Shape:
                    shape = (1, 8)

                return {"input_ids": _Shape()}

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        captured = {}

        def fake_install_attention(model, wrapper_cls, tracker, bits, payload, modules):
            tracker["kivi_quantized_layers"] = 1
            tracker["kivi_kernel_calls"] = 1

        def fake_generate_decode(model, tokenizer, device, payload, torch, past_key_values=None):
            captured["past_key_values"] = past_key_values
            return {"ok": True, "measured": True, "output_text": "ok", "latency_ms": 1.0, "ttft_ms": 1.0, "peak_memory_mib": 1.0, "resident_memory_mib": 1.0}

        built_cache = KIVICache(3, residual_length=32, group_size=32, k_bits=4, v_bits=4)

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(
                qwen2_kv_runtime,
                "_load_qwen2_model",
                return_value=(
                    SimpleNamespace(
                        config=SimpleNamespace(
                            num_hidden_layers=3,
                            model_type="qwen2",
                            num_attention_heads=2,
                            num_key_value_heads=1,
                        ),
                        model=SimpleNamespace(
                            layers=[
                                SimpleNamespace(
                                    self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                                )
                            ]
                        ),
                    ),
                    FakeTokenizer(),
                    "cuda:0",
                ),
            ),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention", side_effect=fake_install_attention),
            patch.object(qwen2_kv_runtime, "build_kivi_cache", return_value=built_cache) as build_cache_mock,
            patch.object(qwen2_kv_runtime, "_generate_decode", side_effect=fake_generate_decode),
        ):
            result = qwen2_kv_runtime._run_kivi_profile(
                {
                    "profile": "kivi_4bit_residual32",
                    "prompt": "prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 2,
                    "kivi_residual_length": 32,
                    "kivi_group_size": 64,
                    "bits": 4,
                }
            )

        self.assertTrue(result["ok"])
        build_cache_mock.assert_called_once()
        self.assertIs(captured["past_key_values"], built_cache)
        self.assertNotIn("runtime_cache", result)
        self.assertNotIn("past_key_values", result)

    def test_run_h2o_profile_seeds_generate_with_h2o_cache(self) -> None:
        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                class _Shape:
                    shape = (1, 20)

                return {"input_ids": _Shape()}

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        captured = {}
        tracker_ref = {}

        def fake_install_attention(model, wrapper_cls, tracker, bits, payload, modules):
            tracker_ref["tracker"] = tracker

        def fake_generate_decode(model, tokenizer, device, payload, torch, past_key_values=None, **kwargs):
            captured["past_key_values"] = past_key_values
            tracker_ref["tracker"]["h2o_prune_events"] = 1
            tracker_ref["tracker"]["h2o_kept_tokens"] = 4
            return {"ok": True, "measured": True, "output_text": "ok", "latency_ms": 1.0, "ttft_ms": 1.0, "peak_memory_mib": 1.0, "resident_memory_mib": 1.0}

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(
                qwen2_kv_runtime,
                "_load_qwen2_model",
                return_value=(
                    SimpleNamespace(
                        config=SimpleNamespace(
                            num_hidden_layers=3,
                            model_type="qwen2",
                            num_attention_heads=2,
                            num_key_value_heads=1,
                        ),
                        model=SimpleNamespace(
                            layers=[
                                SimpleNamespace(
                                    self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                                )
                            ]
                        ),
                    ),
                    FakeTokenizer(),
                    "cuda:0",
                ),
            ),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention", side_effect=fake_install_attention),
            patch.object(qwen2_kv_runtime, "_generate_decode", side_effect=fake_generate_decode),
        ):
            result = qwen2_kv_runtime._run_h2o_profile(
                {
                    "profile": "h2o_heavy10_recent10",
                    "prompt": "prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 2,
                    "h2o_heavy_ratio": 0.1,
                    "h2o_recent_ratio": 0.1,
                }
            )

        self.assertTrue(result["ok"])
        self.assertIsInstance(captured["past_key_values"], H2OCache)
        self.assertEqual(captured["past_key_values"].cache_budget, 4)
        self.assertEqual(result["h2o_prune_events"], 1)
        self.assertEqual(result["h2o_cache_budget"], 4)
        self.assertEqual(result["h2o_kept_tokens"], 4)
        self.assertEqual(result["h2o_prompt_tokens"], 20)

    def test_run_h2o_profile_keeps_short_prompt_when_decode_triggers_pruning(self) -> None:
        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                class _Shape:
                    shape = (1, 1)

                return {"input_ids": _Shape()}

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        tracker_ref = {}

        def fake_install_attention(model, wrapper_cls, tracker, bits, payload, modules):
            tracker_ref["tracker"] = tracker

        def fake_generate_decode(model, tokenizer, device, payload, torch, past_key_values=None, **kwargs):
            tracker_ref["tracker"]["h2o_prune_events"] = 3
            tracker_ref["tracker"]["h2o_kept_tokens"] = tracker_ref["tracker"]["h2o_cache_budget"]
            return {
                "ok": True,
                "measured": True,
                "output_text": "continued",
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 1.0,
                "resident_memory_mib": 1.0,
            }

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(
                qwen2_kv_runtime,
                "_load_qwen2_model",
                return_value=(
                    SimpleNamespace(
                        config=SimpleNamespace(
                            num_hidden_layers=3,
                            model_type="qwen2",
                            num_attention_heads=2,
                            num_key_value_heads=1,
                        ),
                        model=SimpleNamespace(
                            layers=[
                                SimpleNamespace(
                                    self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                                )
                            ]
                        ),
                    ),
                    FakeTokenizer(),
                    "cuda:0",
                ),
            ),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention", side_effect=fake_install_attention),
            patch.object(qwen2_kv_runtime, "_generate_decode", side_effect=fake_generate_decode),
        ):
            result = qwen2_kv_runtime._run_h2o_profile(
                {
                    "profile": "h2o_heavy10_recent10",
                    "prompt": "continue",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 16,
                    "h2o_heavy_ratio": 0.1,
                    "h2o_recent_ratio": 0.1,
                }
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["measured"])
        self.assertEqual(result["h2o_prompt_tokens"], 1)
        self.assertEqual(result["h2o_cache_budget"], 2)
        self.assertEqual(result["h2o_prune_events"], 3)

    def test_run_kivi_profile_keeps_short_unquantized_request_as_success(self) -> None:
        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                class _Shape:
                    shape = (1, 8)

                return {"input_ids": _Shape()}

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        def fake_generate_decode(model, tokenizer, device, payload, torch, past_key_values=None, **kwargs):
            return {
                "ok": True,
                "measured": True,
                "output_text": "short result",
                "latency_ms": 2.0,
                "ttft_ms": 2.0,
                "peak_memory_mib": 3.0,
                "resident_memory_mib": 4.0,
            }

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(
                qwen2_kv_runtime,
                "_load_qwen2_model",
                return_value=(
                    SimpleNamespace(
                        config=SimpleNamespace(
                            num_hidden_layers=2,
                            model_type="qwen2",
                            num_attention_heads=2,
                            num_key_value_heads=1,
                        ),
                        model=SimpleNamespace(
                            layers=[
                                SimpleNamespace(
                                    self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                                )
                            ]
                        ),
                    ),
                    FakeTokenizer(),
                    "cuda:0",
                ),
            ),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention"),
            patch.object(qwen2_kv_runtime, "build_kivi_cache", return_value=KIVICache(2, residual_length=64, group_size=32, k_bits=4, v_bits=4)),
            patch.object(qwen2_kv_runtime, "_generate_decode", side_effect=fake_generate_decode),
        ):
            result = qwen2_kv_runtime._run_kivi_profile(
                {
                    "profile": "kivi_4bit_residual64",
                    "prompt": "short prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 16,
                    "kivi_residual_length": 64,
                    "bits": 4,
                }
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["measured"])
        self.assertFalse(result["kivi_quantization_triggered"])
        self.assertEqual(result["kivi_effective_mode"], "unquantized_short_request")
        self.assertEqual(result["kivi_quantized_layers"], 0)
        self.assertEqual(result["kivi_kernel_calls"], 0)

    def test_run_kivi_profile_marks_quantized_requests_explicitly(self) -> None:
        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                class _Shape:
                    shape = (1, 80)

                return {"input_ids": _Shape()}

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        def fake_install_attention(model, wrapper_cls, tracker, bits, payload, modules):
            tracker["kivi_quantized_layers"] = 2
            tracker["kivi_kernel_calls"] = 5
            tracker["kivi_quantize_calls"] = 2
            tracker["kivi_quantized_tokens"] = 64

        def fake_generate_decode(model, tokenizer, device, payload, torch, past_key_values=None, **kwargs):
            return {
                "ok": True,
                "measured": True,
                "output_text": "quantized result",
                "latency_ms": 2.0,
                "ttft_ms": 2.0,
                "peak_memory_mib": 3.0,
                "resident_memory_mib": 4.0,
            }

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(
                qwen2_kv_runtime,
                "_load_qwen2_model",
                return_value=(
                    SimpleNamespace(
                        config=SimpleNamespace(
                            num_hidden_layers=2,
                            model_type="qwen2",
                            num_attention_heads=2,
                            num_key_value_heads=1,
                        ),
                        model=SimpleNamespace(
                            layers=[
                                SimpleNamespace(
                                    self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                                )
                            ]
                        ),
                    ),
                    FakeTokenizer(),
                    "cuda:0",
                ),
            ),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention", side_effect=fake_install_attention),
            patch.object(qwen2_kv_runtime, "build_kivi_cache", return_value=KIVICache(2, residual_length=32, group_size=32, k_bits=4, v_bits=4)),
            patch.object(qwen2_kv_runtime, "_generate_decode", side_effect=fake_generate_decode),
        ):
            result = qwen2_kv_runtime._run_kivi_profile(
                {
                    "profile": "kivi_4bit_residual32",
                    "prompt": "long prompt",
                    "model_name": "/tmp/model",
                    "max_new_tokens": 16,
                    "kivi_residual_length": 32,
                    "bits": 4,
                }
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["kivi_quantization_triggered"])
        self.assertEqual(result["kivi_effective_mode"], "quantized")

    def test_qwen2_kivi_attention_reads_and_updates_kivi_cache(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda:0"
                self.dtype = "float16"

            def size(self):
                return self.shape

            def view(self, *shape):
                return FakeTensor(shape)

            def transpose(self, dim0, dim1):
                shape = list(self.shape)
                shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                return FakeTensor(tuple(shape))

            def contiguous(self):
                return self

            def reshape(self, *shape):
                return FakeTensor(shape)

            def __getitem__(self, item):
                if not isinstance(item, tuple):
                    return self
                shape = list(self.shape)
                for dim, selector in enumerate(item):
                    if isinstance(selector, slice):
                        start = 0 if selector.start is None else selector.start
                        stop = shape[dim] if selector.stop is None else selector.stop
                        if start < 0:
                            start += shape[dim]
                        if stop < 0:
                            stop += shape[dim]
                        shape[dim] = max(0, stop - start)
                return FakeTensor(tuple(shape))

            def __truediv__(self, other):
                return self

            def to(self, _dtype):
                return self

        class FakeModule:
            def __call__(self, value):
                return FakeTensor(value.shape)

        class FakeNN:
            class Module:
                def __init__(self):
                    pass

            functional = SimpleNamespace(softmax=lambda attn_weights, dim=-1, dtype=None: attn_weights)

        class FakeNoopContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeCuda:
            @staticmethod
            def device(_device):
                return FakeNoopContext()

        class FakeTorch:
            cuda = FakeCuda()
            float32 = "float32"

            @staticmethod
            def matmul(left, right):
                if len(left.shape) == 4 and len(right.shape) == 4:
                    return FakeTensor((left.shape[0], left.shape[1], left.shape[2], right.shape[3]))
                return FakeTensor(left.shape)

            @staticmethod
            def cat(tensors, dim=0):
                shape = list(tensors[0].shape)
                shape[dim] = sum(t.shape[dim] for t in tensors)
                return FakeTensor(tuple(shape))

        source = SimpleNamespace(
            q_proj=FakeModule(),
            k_proj=FakeModule(),
            v_proj=FakeModule(),
            o_proj=FakeModule(),
            head_dim=4,
            rotary_emb=None,
        )
        config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2)
        tracker = {"kivi_kernel_calls": 0, "kivi_quantize_calls": 0, "kivi_quantized_layers": 0, "kivi_quantized_tokens": 0}
        modules = {
            "torch": FakeTorch(),
            "F": object(),
            "nn": FakeNN(),
            "repeat_kv": lambda tensor, groups: FakeTensor((tensor.shape[0], 2, tensor.shape[2], tensor.shape[3])),
            "cuda_bmm_fA_qB_outer": lambda *args, **kwargs: FakeTensor((1, 2, 1, 1)),
            "triton_quantize_and_pack_along_last_dim": lambda tensor, group_size, bits: (
                FakeTensor((1, 1, 4, 1)),
                FakeTensor((1, 1, 1, 1)),
                FakeTensor((1, 1, 1, 1)),
            ),
            "apply_rotary_pos_emb": lambda q, k, *args: (q, k),
        }
        attention = qwen2_kv_runtime.Qwen2KIVIAttention(
            source,
            config,
            1,
            tracker,
            4,
            {"kivi_group_size": 32, "kivi_residual_length": 32},
            modules,
        )
        cache = KIVICache(2, residual_length=32, group_size=32, k_bits=4, v_bits=4)
        cache.update_quantized(1, qwen2_kv_runtime.KIVILayerState(None, FakeTensor((1, 1, 1, 4)), None, None, None, FakeTensor((1, 1, 1, 4)), None, None, 1))

        attn_output, _, past = attention.forward(
            FakeTensor((1, 1, 8)),
            attention_mask=None,
            position_ids=None,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
        )

        self.assertIs(past, cache)
        self.assertIs(cache[1], past[1])
        self.assertEqual(cache.get_seq_length(1), 2)
        self.assertEqual(attn_output.shape, (1, 1, 8))

    def test_qwen2_kivi_attention_rejects_legacy_tuple_cache(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda:0"
                self.dtype = "float16"

            def size(self):
                return self.shape

            def view(self, *shape):
                return FakeTensor(shape)

            def transpose(self, dim0, dim1):
                shape = list(self.shape)
                shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                return FakeTensor(tuple(shape))

            def contiguous(self):
                return self

        class FakeModule:
            def __call__(self, value):
                return FakeTensor(value.shape)

        class FakeNN:
            class Module:
                def __init__(self):
                    pass

            functional = SimpleNamespace(softmax=lambda attn_weights, dim=-1, dtype=None: attn_weights)

        source = SimpleNamespace(
            q_proj=FakeModule(),
            k_proj=FakeModule(),
            v_proj=FakeModule(),
            o_proj=FakeModule(),
            head_dim=4,
            rotary_emb=None,
        )
        config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1)
        tracker = {"kivi_kernel_calls": 0, "kivi_quantize_calls": 0, "kivi_quantized_layers": 0, "kivi_quantized_tokens": 0}
        modules = {
            "torch": object(),
            "F": object(),
            "nn": FakeNN(),
            "repeat_kv": lambda tensor, groups: tensor,
            "cuda_bmm_fA_qB_outer": lambda *args, **kwargs: None,
            "triton_quantize_and_pack_along_last_dim": lambda tensor, group_size, bits: (None, None, None),
            "apply_rotary_pos_emb": lambda q, k, *args: (q, k),
        }
        attention = qwen2_kv_runtime.Qwen2KIVIAttention(
            source,
            config,
            0,
            tracker,
            4,
            {"kivi_group_size": 32, "kivi_residual_length": 32},
            modules,
        )

        with self.assertRaisesRegex(TypeError, "KIVICache"):
            attention.forward(
                FakeTensor((1, 1, 8)),
                attention_mask=None,
                position_ids=None,
                past_key_value=((None, None),),
                output_attentions=False,
                use_cache=True,
            )

    def test_qwen2_h2o_attention_rejects_legacy_tuple_cache(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda:0"
                self.dtype = "float16"

            def size(self):
                return self.shape

            def view(self, *shape):
                return FakeTensor(shape)

            def transpose(self, dim0, dim1):
                shape = list(self.shape)
                shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                return FakeTensor(tuple(shape))

        class FakeModule:
            def __call__(self, value):
                return FakeTensor(value.shape)

        class FakeNN:
            class Module:
                def __init__(self):
                    pass

            functional = SimpleNamespace(softmax=lambda attn_weights, dim=-1, dtype=None: attn_weights)

        source = SimpleNamespace(
            q_proj=FakeModule(),
            k_proj=FakeModule(),
            v_proj=FakeModule(),
            o_proj=FakeModule(),
            head_dim=4,
            rotary_emb=None,
        )
        config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1)
        tracker = {"h2o_prune_events": 0, "h2o_mask_events": 0, "h2o_cache_budget": 0, "h2o_kept_tokens": 0, "h2o_prompt_tokens": 0}
        modules = {
            "torch": object(),
            "F": object(),
            "nn": FakeNN(),
            "repeat_kv": lambda tensor, groups: tensor,
            "apply_rotary_pos_emb": lambda q, k, *args: (q, k),
        }
        attention = qwen2_kv_runtime.Qwen2H2OAttention(
            source,
            config,
            0,
            tracker,
            0,
            {"h2o_heavy_size": 2, "h2o_recent_size": 1},
            modules,
        )

        with self.assertRaisesRegex(TypeError, "H2OCache"):
            attention.forward(
                FakeTensor((1, 1, 8)),
                attention_mask=None,
                position_ids=None,
                past_key_value=((None, None),),
                output_attentions=False,
                use_cache=True,
            )

    def test_qwen2_kivi_attention_prefill_quantizes_full_prompt_before_residual_tail(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda:0"
                self.dtype = "float16"

            def size(self):
                return self.shape

            def view(self, *shape):
                return FakeTensor(shape)

            def transpose(self, dim0, dim1):
                shape = list(self.shape)
                shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                return FakeTensor(tuple(shape))

            def contiguous(self):
                return self

            def reshape(self, *shape):
                return FakeTensor(shape)

            def __getitem__(self, item):
                if not isinstance(item, tuple):
                    return self
                shape = list(self.shape)
                for dim, selector in enumerate(item):
                    if isinstance(selector, slice):
                        start = 0 if selector.start is None else selector.start
                        stop = shape[dim] if selector.stop is None else selector.stop
                        if start < 0:
                            start += shape[dim]
                        if stop < 0:
                            stop += shape[dim]
                        shape[dim] = max(0, stop - start)
                return FakeTensor(tuple(shape))

            def __truediv__(self, other):
                return self

            def to(self, _dtype):
                return self

        class FakeModule:
            def __call__(self, value):
                return FakeTensor(value.shape)

        class FakeNN:
            class Module:
                def __init__(self):
                    pass

            functional = SimpleNamespace(softmax=lambda attn_weights, dim=-1, dtype=None: attn_weights)

        class FakeTorch:
            class cuda:
                @staticmethod
                def device(_device):
                    class _Context:
                        def __enter__(self):
                            return None

                        def __exit__(self, exc_type, exc, tb):
                            return False

                    return _Context()

            float32 = "float32"

            @staticmethod
            def matmul(left, right):
                return FakeTensor((left.shape[0], left.shape[1], left.shape[2], right.shape[3]))

            @staticmethod
            def cat(tensors, dim=0):
                shape = list(tensors[0].shape)
                shape[dim] = sum(t.shape[dim] for t in tensors)
                return FakeTensor(tuple(shape))

        quant_inputs = {"shapes": []}

        def fake_quant_pack(tensor, group_size, bits):
            quant_inputs["shapes"].append(tensor.shape)
            return (
                FakeTensor((1, 1, 4, 42)),
                FakeTensor((1, 1, 1, 42)),
                FakeTensor((1, 1, 1, 42)),
            )

        source = SimpleNamespace(
            q_proj=FakeModule(),
            k_proj=FakeModule(),
            v_proj=FakeModule(),
            o_proj=FakeModule(),
            head_dim=4,
            rotary_emb=None,
        )
        config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2)
        tracker = {"kivi_kernel_calls": 0, "kivi_quantize_calls": 0, "kivi_quantized_layers": 0, "kivi_quantized_tokens": 0}
        modules = {
            "torch": FakeTorch(),
            "F": object(),
            "nn": FakeNN(),
            "repeat_kv": lambda tensor, groups: FakeTensor((tensor.shape[0], 2, tensor.shape[2], tensor.shape[3])),
            "cuda_bmm_fA_qB_outer": lambda *args, **kwargs: FakeTensor((1, 2, 1, 42)),
            "triton_quantize_and_pack_along_last_dim": fake_quant_pack,
            "apply_rotary_pos_emb": lambda q, k, *args: (q, k),
        }
        attention = qwen2_kv_runtime.Qwen2KIVIAttention(
            source,
            config,
            0,
            tracker,
            4,
            {"kivi_group_size": 32, "kivi_residual_length": 32},
            modules,
        )

        attn_output, _, past = attention.forward(
            FakeTensor((1, 1353, 8)),
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        )

        self.assertEqual(quant_inputs["shapes"][0], (1, 1, 4, 1344))
        self.assertEqual(past[0].key_full.shape, (1, 1, 9, 4))
        self.assertEqual(past[0].value_full.shape, (1, 1, 32, 4))
        self.assertEqual(past[0].kv_seq_len, 1353)
        self.assertEqual(attn_output.shape, (1, 1353, 8))

    def test_qwen2_runtime_batch_stops_after_fatal_cuda_assert(self) -> None:
        calls = []

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi"}

        def fake_run_request(runtime, request, worker_mode="batch"):
            calls.append(request["request_id"])
            if request["request_id"] == "r1":
                return {
                    "ok": False,
                    "measured": False,
                    "error": "RuntimeError: CUDA error: device-side assert triggered",
                    "failure_stage": "generate",
                    "worker_mode": "batch",
                }
            raise AssertionError("later requests must not run after fatal CUDA assert")

        payload = {
            "requests": [
                {"request_id": "r1", "profile": "kivi_4bit_residual32", "prompt": "a"},
                {"request_id": "r2", "profile": "kivi_4bit_residual32", "prompt": "b"},
                {"request_id": "r3", "profile": "kivi_4bit_residual32", "prompt": "c"},
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertEqual(calls, ["r1"])
        self.assertEqual(len(result["results"]), 3)
        self.assertTrue(all("device-side assert triggered" in row["error"] for row in result["results"]))

    def test_qwen2_runtime_batch_reuses_kivi_cache_for_same_session_prefix_extension(self) -> None:
        caches = [object(), object()]
        past_caches = []
        prompts = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __init__(self):
                self.vocab = {"hello": 1, "world": 2}

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor([self.vocab[token] for token in prompt.split()])}

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi", "tokenizer": FakeTokenizer(), "model": SimpleNamespace(config=SimpleNamespace())}

        def fake_run_request(runtime, request, worker_mode="batch"):
            prompts.append(request["prompt"])
            past_caches.append(request.get("_runtime_reusable_kivi_cache"))
            prompt_token_ids = [runtime["tokenizer"].vocab[token] for token in request["prompt"].split()]
            return {
                "ok": True,
                "measured": True,
                "output_text": request["prompt"],
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 2.0,
                "kv_cache_memory_mib": 2.0,
                "runtime_cache": caches[len(prompts) - 1],
                "runtime_prompt_token_ids": prompt_token_ids,
            }

        payload = {
            "requests": [
                {"request_id": "r1", "session_id": "s1", "turn_index": 0, "profile": "kivi_4bit_residual32", "prompt": "hello"},
                {"request_id": "r2", "session_id": "s1", "turn_index": 1, "profile": "kivi_4bit_residual32", "prompt": "hello world"},
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
            patch.object(qwen2_kv_runtime, "_release_runtime_resources"),
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertTrue(result["ok"])
        self.assertEqual(past_caches, [None, caches[0]])
        self.assertNotIn("runtime_cache", result["results"][0])
        self.assertNotIn("runtime_prompt_token_ids", result["results"][0])

    def test_qwen2_runtime_batch_rebuilds_kivi_cache_when_prompt_is_not_prefix_extension(self) -> None:
        rebuild_reasons = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __init__(self):
                self.vocab = {"hello": 1, "there": 2, "different": 3, "path": 4}

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor([self.vocab[token] for token in prompt.split()])}

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi", "tokenizer": FakeTokenizer(), "model": SimpleNamespace(config=SimpleNamespace())}

        def fake_run_request(runtime, request, worker_mode="batch"):
            rebuild_reasons.append(request.get("_runtime_cache_rebuild_reason"))
            return {
                "ok": True,
                "measured": True,
                "output_text": request["prompt"],
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 2.0,
                "kv_cache_memory_mib": 2.0,
                "runtime_cache": object(),
                "runtime_prompt_token_ids": request["prompt"].split(),
            }

        payload = {
            "requests": [
                {"request_id": "r1", "session_id": "s1", "turn_index": 0, "profile": "kivi_4bit_residual32", "prompt": "hello there"},
                {"request_id": "r2", "session_id": "s1", "turn_index": 1, "profile": "kivi_4bit_residual32", "prompt": "different path"},
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
            patch.object(qwen2_kv_runtime, "_release_runtime_resources"),
        ):
            qwen2_kv_runtime.run_profile_batch(payload)

        self.assertEqual(rebuild_reasons, ["new_session", "prompt_mismatch"])

    def test_qwen2_runtime_batch_does_not_reuse_kivi_cache_when_history_turns_are_present(self) -> None:
        past_caches = []
        rebuild_reasons = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __init__(self):
                self.vocab = {"hello": 1, "user:": 2, "hi": 3, "assistant:": 4, "again": 5}

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor([self.vocab[token] for token in prompt.lower().split()])}

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi", "tokenizer": FakeTokenizer(), "model": SimpleNamespace(config=SimpleNamespace())}

        def fake_run_request(runtime, request, worker_mode="batch"):
            past_caches.append(request.get("_runtime_reusable_kivi_cache"))
            rebuild_reasons.append(request.get("_runtime_cache_rebuild_reason"))
            return {
                "ok": True,
                "measured": True,
                "output_text": request["prompt"],
                "latency_ms": 1.0,
                "ttft_ms": 1.0,
                "peak_memory_mib": 2.0,
                "kv_cache_memory_mib": 2.0,
                "runtime_cache": object(),
                "runtime_prompt_token_ids": request["prompt"].lower().split(),
            }

        payload = {
            "requests": [
                {"request_id": "r1", "session_id": "s1", "turn_index": 0, "profile": "kivi_4bit_residual32", "prompt": "hello", "history_turns": []},
                {
                    "request_id": "r2",
                    "session_id": "s1",
                    "turn_index": 1,
                    "profile": "kivi_4bit_residual32",
                    "prompt": "User: hello\nAssistant: hi\nUser: again",
                    "history_turns": ["User: hello", "Assistant: hi"],
                },
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
            patch.object(qwen2_kv_runtime, "_release_runtime_resources"),
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertTrue(result["ok"])
        self.assertEqual(past_caches, [None, None])
        self.assertEqual(rebuild_reasons, ["new_session", "history_turns_present"])

    def test_attach_kivi_session_cache_rebuilds_when_profile_changes(self) -> None:
        rebuild_reasons = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __init__(self):
                self.vocab = {"hello": 1, "world": 2}

            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor([self.vocab[token] for token in prompt.split()])}

        runtime = {
            "tokenizer": FakeTokenizer(),
            "session_reuse": {
                "s1": {
                    "profile": "kivi_4bit_residual32",
                    "cache": object(),
                    "prompt_token_ids": ["hello"],
                }
            },
        }
        request = qwen2_kv_runtime._attach_kivi_session_cache(
            runtime,
            {"request_id": "r2", "session_id": "s1", "turn_index": 1, "profile": "kivi_2bit_residual32", "prompt": "hello world"},
        )

        rebuild_reasons.append(request.get("_runtime_cache_rebuild_reason"))
        self.assertEqual(rebuild_reasons, ["profile_changed"])

    def test_qwen2_runtime_batch_clears_runtime_resources_after_oom(self) -> None:
        cleanup_calls = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor(prompt.split())}

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi", "tokenizer": FakeTokenizer(), "model": SimpleNamespace(config=SimpleNamespace())}

        def fake_run_request(runtime, request, worker_mode="batch"):
            if request["request_id"] == "r1":
                return {
                    "ok": False,
                    "measured": False,
                    "error": "CUDA out of memory",
                    "failure_stage": "generate",
                    "worker_mode": "batch",
                }
            raise AssertionError("later requests must not run after OOM")

        payload = {
            "requests": [
                {"request_id": "r1", "session_id": "s1", "turn_index": 0, "profile": "kivi_4bit_residual32", "prompt": "a"},
                {"request_id": "r2", "session_id": "s1", "turn_index": 1, "profile": "kivi_4bit_residual32", "prompt": "a b"},
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
            patch.object(qwen2_kv_runtime, "_release_runtime_resources", side_effect=lambda runtime: cleanup_calls.append(runtime)),
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertFalse(result["ok"])
        self.assertEqual(len(cleanup_calls), 1)

    def test_qwen2_runtime_batch_clears_runtime_resources_when_request_raises(self) -> None:
        cleanup_calls = []

        class FakeTensor:
            def __init__(self, values):
                self._values = [values]

            def tolist(self):
                return self._values

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                return {"input_ids": FakeTensor(prompt.split())}

        def fake_prepare(request, worker_start):
            return {"runtime": "kivi", "tokenizer": FakeTokenizer(), "model": SimpleNamespace(config=SimpleNamespace())}

        def fake_run_request(runtime, request, worker_mode="batch"):
            raise RuntimeError("boom")

        payload = {
            "requests": [
                {"request_id": "r1", "session_id": "s1", "turn_index": 0, "profile": "kivi_4bit_residual32", "prompt": "a"},
                {"request_id": "r2", "session_id": "s1", "turn_index": 1, "profile": "kivi_4bit_residual32", "prompt": "a b"},
            ]
        }

        with (
            patch.object(qwen2_kv_runtime, "_prepare_kivi_runtime", side_effect=fake_prepare),
            patch.object(qwen2_kv_runtime, "_run_kivi_request", side_effect=fake_run_request),
            patch.object(qwen2_kv_runtime, "_release_runtime_resources", side_effect=lambda runtime: cleanup_calls.append(runtime)),
        ):
            result = qwen2_kv_runtime.run_profile_batch(payload)

        self.assertFalse(result["ok"])
        self.assertEqual(len(cleanup_calls), 1)

    def test_prepare_kivi_runtime_rejects_unverified_attention_layout(self) -> None:
        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        bad_model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=2,
                model_type="unknown_decoder",
                num_attention_heads=8,
                num_key_value_heads=2,
            ),
            model=SimpleNamespace(layers=[]),
        )

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(qwen2_kv_runtime, "_load_qwen2_model", return_value=(bad_model, object(), "cuda:0")),
        ):
            with self.assertRaisesRegex(ValueError, "unsupported KIVI runtime model"):
                qwen2_kv_runtime._prepare_kivi_runtime(
                    {"profile": "kivi_4bit_residual32", "model_name": "/tmp/model"},
                    worker_start=0.0,
                )

    def test_prepare_kivi_runtime_accepts_qwen2_layout(self) -> None:
        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

        qwen2_model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=2,
                model_type="qwen2",
                num_attention_heads=8,
                num_key_value_heads=2,
                use_cache=False,
            ),
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                    ),
                    SimpleNamespace(
                        self_attn=SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object(), o_proj=object())
                    ),
                ]
            ),
        )

        with (
            patch.object(qwen2_kv_runtime, "_import_runtime_modules", return_value={"torch": FakeTorch(), "AutoModelForCausalLM": object(), "AutoTokenizer": object()}),
            patch.object(qwen2_kv_runtime, "_require_cuda"),
            patch.object(qwen2_kv_runtime, "_load_qwen2_model", return_value=(qwen2_model, object(), "cuda:0")),
            patch.object(qwen2_kv_runtime, "_install_qwen2_attention") as install_attention,
        ):
            runtime = qwen2_kv_runtime._prepare_kivi_runtime(
                {"profile": "kivi_4bit_residual32", "model_name": "/tmp/model"},
                worker_start=0.0,
            )

        self.assertIs(runtime["model"], qwen2_model)
        install_attention.assert_called_once()

    def test_kivi_failure_message_reports_prompt_tokens(self) -> None:
        error = (
            "KIVI proof missing: no quantized cache block and/or quant GEMV kernel call was observed. "
            "prompt_tokens=9 max_new_tokens=16 residual_length=32 "
            "quantized_layers=0 kernel_calls=0"
        )

        self.assertIn("prompt_tokens=9", error)
        self.assertIn("residual_length=32", error)

    def test_kivi_proof_error_includes_token_context(self) -> None:
        message = qwen2_kv_runtime._kivi_proof_error(
            prompt_tokens=9,
            max_new_tokens=16,
            residual_length=32,
            quantized_layers=0,
            kernel_calls=0,
        )

        self.assertIn("prompt_tokens=9", message)
        self.assertIn("max_new_tokens=16", message)
        self.assertIn("residual_length=32", message)
        self.assertIn("quantized_layers=0", message)
        self.assertIn("kernel_calls=0", message)

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

    def test_conformal_guard_counts_configured_lossy_profiles_without_samples(self) -> None:
        guard = ConformalGuard(
            epsilon=0.5,
            delta=0.2,
            exact_profiles={"full_gpu"},
            profiles=["full_gpu", "kivi_4bit", "kivi_2bit", "h2o_heavy_hitter"],
            calibration_rows=[
                _measurement("r1", "kivi_4bit", 0.1),
                _measurement("r2", "kivi_4bit", 0.2),
            ],
        )
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit", "kivi_2bit", "h2o_heavy_hitter"})
        self.assertAlmostEqual(guard.delta_a, 0.2 / 3)

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

    def test_conformal_guard_empty_calibration_set_makes_lossy_profiles_unsafe(self) -> None:
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"full_gpu"},
            profiles=["full_gpu", "kivi_4bit"],
            calibration_rows=[],
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit"})
        self.assertEqual(guard.risk_upper(request, "full_gpu", 0.9), 0.0)
        self.assertFalse(guard.is_safe(request, "kivi_4bit", 0.0))

    def test_conformal_guard_builds_large_calibration_table_without_quadratic_regression(self) -> None:
        rows = [
            _measurement(f"r{i}", "kivi_4bit" if i % 2 == 0 else "h2o_heavy_hitter", (i % 10) / 100.0)
            for i in range(6000)
        ]
        started = time.perf_counter()
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            profiles=["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            exact_profiles={"full_gpu"},
            calibration_rows=rows,
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit", "h2o_heavy_hitter"})

    def test_registry_builds_only_required_august_baselines(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        policies = build_policies(
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(
            [policy.name for policy in policies],
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
        )

    def test_registry_builds_tailguard_and_quality_oracle(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        policies = build_policies(
            ["tailguard", "quality_oracle"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual([policy.name for policy in policies], ["tailguard", "quality_oracle"])
        self.assertFalse(policies[0].placeholder)
        self.assertTrue(policies[1].oracle)

    def test_tailguard_selects_fastest_calibrated_safe_lossy_profile(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "kivi_4bit", 0.01, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.01, ttft_ms=6.0),
            _measurement("c1", "h2o_heavy", 0.01, ttft_ms=9.0),
            _measurement("c2", "h2o_heavy", 0.01, ttft_ms=10.0),
        ]
        [policy] = build_policies(["tailguard"], rows, rows, ["full_gpu", "kivi_4bit", "h2o_heavy"], 0.05, 0.05, {"full_gpu"})

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "kivi_4bit")
        self.assertTrue(action.safe)
        self.assertEqual(action.candidate_safe_count, 2.0)
        self.assertIsNotNone(action.controller_qrp_ms)

    def test_tailguard_falls_back_to_full_gpu_and_records_rejected_unsafe(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "kivi_4bit", 0.5, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.5, ttft_ms=6.0),
        ]
        [policy] = build_policies(["tailguard"], rows, rows, ["full_gpu", "kivi_4bit"], 0.05, 0.05, {"full_gpu"}, record_rejected_unsafe=True)

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.rejected_profile, "kivi_4bit")
        self.assertGreater(action.rejected_risk_upper, 0.05)

    def test_quality_oracle_reads_eval_truth_and_marks_summary_oracle(self) -> None:
        rows = [
            _measurement("e1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("e1", "kivi_4bit", 0.04, ttft_ms=5.0),
            _measurement("e1", "h2o_heavy", 0.2, ttft_ms=3.0),
        ]
        [policy] = build_policies(["quality_oracle"], rows, rows, ["full_gpu", "kivi_4bit", "h2o_heavy"], 0.05, 0.05, {"full_gpu"})

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"length_bucket": "short"}), None, None)
        record = PolicyRunRecord(policy.name, "e1", action.profile, True, True, oracle=policy.oracle, quality_loss=0.04)
        summary = MetricCollector().summarize_policy_runs([record], epsilon=0.05, delta=0.05, exact_profiles={"full_gpu"})

        self.assertEqual(action.profile, "kivi_4bit")
        self.assertTrue(summary["quality_oracle"]["oracle"])

    def test_registry_keeps_august_baseline_order_stable(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=20.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=5.0),
        ]
        policies = build_policies(
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(
            [policy.name for policy in policies],
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
        )

    def test_static_baselines_fallback_to_exact_when_no_lossy_profile_is_selectable(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=20.0),
            _measurement("c1", "kivi_4bit", 0.8, ttft_ms=5.0),
        ]
        policies = build_policies(
            ["static_best", "static_safe"],
            rows,
            rows,
            ["kivi_4bit", "full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual([policy.decide(request, None, None).profile for policy in policies], ["full_gpu", "full_gpu"])

    def test_full_lru_fixed_to_full_gpu(self) -> None:
        [policy] = build_policies(
            ["full_lru"],
            [_measurement("c1", "full_gpu", 0.0)],
            [_measurement("e1", "full_gpu", 0.0)],
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt"), None, None)
        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.reason, "full precision exact profile")

    def test_full_lru_uses_engine_full_lru_for_vllm_only_profiles(self) -> None:
        [policy] = build_policies(
            ["full_lru"],
            [_measurement("c1", "engine_full_lru", 0.0)],
            [_measurement("e1", "engine_full_lru", 0.0)],
            ["engine_full_lru"],
            0.2,
            0.05,
            {"engine_full_lru"},
        )
        action = policy.decide(Request("e1", "qa", "prompt"), None, None)
        self.assertEqual(action.profile, "engine_full_lru")

    def test_static_best_selects_lowest_ttft_under_mean_quality_constraint(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=8.0),
            _measurement("c2", "kivi_4bit", 0.1, ttft_ms=9.0),
            _measurement("c1", "kivi_2bit", 0.3, ttft_ms=3.0),
            _measurement("c2", "kivi_2bit", 0.3, ttft_ms=4.0),
        ]
        [policy] = build_policies(
            ["static_best"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "kivi_2bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "kivi_4bit")

    def test_static_safe_rejects_high_violation_rate(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=8.0),
            _measurement("c2", "kivi_4bit", 0.3, ttft_ms=9.0),
            _measurement("c1", "h2o_heavy_hitter", 0.1, ttft_ms=12.0),
            _measurement("c2", "h2o_heavy_hitter", 0.1, ttft_ms=13.0),
        ]
        [policy] = build_policies(
            ["static_safe"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "h2o_heavy_hitter")

    def test_static_safe_falls_back_to_exact_when_fixed_profile_is_request_unsafe(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.01, ttft_ms=8.0),
            _measurement("c2", "kivi_4bit", 0.09, ttft_ms=9.0),
        ]
        [policy] = build_policies(
            ["static_safe"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit"],
            0.05,
            0.6,
            {"full_gpu"},
        )

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.fallback_reason, "calibrated unsafe")

    def test_utility_dynamic_uses_configured_utility_weights(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0, peak_memory_mib=10.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=5.0, peak_memory_mib=100.0),
            _measurement("c2", "kivi_4bit", 0.1, ttft_ms=6.0, peak_memory_mib=100.0),
            _measurement("c1", "h2o_heavy_hitter", 0.12, ttft_ms=8.0, peak_memory_mib=10.0),
            _measurement("c2", "h2o_heavy_hitter", 0.12, ttft_ms=9.0, peak_memory_mib=10.0),
        ]
        [policy] = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 1.0, "loss_weight": 1.0}],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(policy.decide(request, None, None).profile, "h2o_heavy_hitter")

    def test_uncalibrated_dynamic_uses_point_prediction_threshold_only(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.3, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.3, ttft_ms=6.0),
            _measurement("c1", "kivi_2bit", 0.1, ttft_ms=3.0),
            _measurement("c2", "kivi_2bit", 0.1, ttft_ms=4.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_2bit", "kivi_4bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        action = policy.decide(request, None, None)
        self.assertEqual(action.profile, "kivi_2bit")
        self.assertEqual(action.fallback_reason, "")

    def test_uncalibrated_dynamic_uses_stable_sort_not_profile_order(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0, peak_memory_mib=100.0),
            _measurement("c1", "kivi_4bit", 0.04, ttft_ms=5.0, peak_memory_mib=15.0),
            _measurement("c2", "kivi_4bit", 0.04, ttft_ms=6.0, peak_memory_mib=15.0),
            _measurement("c1", "h2o_heavy_hitter", 0.02, ttft_ms=10.0, peak_memory_mib=20.0),
            _measurement("c2", "h2o_heavy_hitter", 0.02, ttft_ms=11.0, peak_memory_mib=20.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.05,
            0.05,
            {"full_gpu"},
        )

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "h2o_heavy_hitter")


    def test_utility_dynamic_uses_lossy_candidates_before_exact_fallback(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=1.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=1.0),
            _measurement("c1", "kivi_4bit", 0.01, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.01, ttft_ms=5.0),
            _measurement("c1", "h2o_heavy", 0.01, ttft_ms=6.0),
            _measurement("c2", "h2o_heavy", 0.01, ttft_ms=6.0),
        ]
        [policy] = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 0.0, "loss_weight": 0.0}],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy"],
            0.2,
            0.05,
            {"full_gpu"},
        )

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "kivi_4bit")

    def test_uncalibrated_dynamic_uses_lossy_candidates_before_exact_fallback(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=1.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=1.0),
            _measurement("c1", "kivi_4bit", 0.01, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.01, ttft_ms=5.0),
            _measurement("c1", "h2o_heavy", 0.01, ttft_ms=6.0),
            _measurement("c2", "h2o_heavy", 0.01, ttft_ms=6.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy"],
            0.2,
            0.05,
            {"full_gpu"},
        )

        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)

        self.assertEqual(action.profile, "kivi_4bit")

    def test_utility_dynamic_reports_unsafe_lossy_without_conformal_fallback(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
            _measurement("c2", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
        ]
        [policy] = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 0.0, "loss_weight": 0.0}],
            rows,
            rows,
            ["full_gpu", "kivi_4bit_residual32"],
            0.2,
            0.05,
            {"full_gpu"},
            record_rejected_unsafe=True,
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "kivi_4bit_residual32")
        self.assertFalse(action.safe)
        self.assertEqual(action.rejected_profile, "")
        self.assertGreater(action.risk_upper or 0.0, 0.2)

    def test_uncalibrated_dynamic_does_not_apply_conformal_fallback_after_point_threshold(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "kivi_4bit_residual32", 0.0, ttft_ms=1.0),
            _measurement("c2", "kivi_4bit_residual32", 0.4, ttft_ms=1.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit_residual32"],
            0.25,
            0.05,
            {"full_gpu"},
            record_rejected_unsafe=True,
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "kivi_4bit_residual32")
        self.assertFalse(action.safe)
        self.assertEqual(action.rejected_profile, "")
        self.assertGreater(action.risk_upper or 0.0, 0.25)

    def test_uncalibrated_dynamic_falls_back_to_full_gpu_when_point_prediction_exceeds_epsilon(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
            _measurement("c2", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit_residual32"],
            0.2,
            0.05,
            {"full_gpu"},
            record_rejected_unsafe=True,
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.rejected_profile, "")

    def test_vllm_lru_rank_tuple_orders_by_recency_then_block_id(self) -> None:
        policy = create_vllm_policy("full_lru")
        pool = SimpleNamespace()
        policy.on_admit(pool, 2)
        policy.on_admit(pool, 1)
        self.assertLess(policy.rank_tuple(pool, 2), policy.rank_tuple(pool, 1))
        self.assertLess(policy.rank_tuple(pool, 3), policy.rank_tuple(pool, 4))

    def test_profile_registry_uses_full_kivi_h2o_adapters(self) -> None:
        adapters = build_profile_adapters(
            ["full", "kivi", "h2o"],
            {"pilot_model": "/tmp/model"},
        )
        self.assertEqual([adapter.name for adapter in adapters], ["full", "kivi", "h2o"])

    def test_profile_registry_exposes_full_gpu_kivi_h2o_grid(self) -> None:
        adapters = build_profile_adapters(
            ["full", "kivi", "h2o"],
            {"pilot_model": "/tmp/model"},
        )
        specs = [spec for adapter in adapters for spec in adapter.profiles()]
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "full_gpu",
                "kivi_4bit_residual16",
                "kivi_4bit_residual32",
                "kivi_4bit_residual64",
                "kivi_2bit_residual16",
                "kivi_2bit_residual32",
                "kivi_2bit_residual64",
                "h2o_heavy05_recent05",
                "h2o_heavy08_recent08",
                "h2o_heavy10_recent10",
                "h2o_heavy15_recent15",
                "h2o_heavy20_recent20",
            ],
        )
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["kivi_4bit_residual16"].metadata["bits"], 4)
        self.assertEqual(by_name["kivi_4bit_residual16"].metadata["kivi_group_size"], 16)
        self.assertEqual(by_name["kivi_4bit_residual16"].metadata["kivi_residual_length"], 16)
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["bits"], 4)
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["kivi_residual_length"], 32)
        self.assertEqual(by_name["kivi_2bit_residual16"].metadata["bits"], 2)
        self.assertEqual(by_name["kivi_2bit_residual16"].metadata["kivi_group_size"], 16)
        self.assertEqual(by_name["kivi_2bit_residual16"].metadata["kivi_residual_length"], 16)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["bits"], 2)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["kivi_residual_length"], 64)
        self.assertEqual(by_name["h2o_heavy05_recent05"].metadata["h2o_heavy_ratio"], 0.05)
        self.assertEqual(by_name["h2o_heavy05_recent05"].metadata["h2o_recent_ratio"], 0.05)
        self.assertEqual(by_name["h2o_heavy08_recent08"].metadata["h2o_heavy_ratio"], 0.08)
        self.assertEqual(by_name["h2o_heavy08_recent08"].metadata["h2o_recent_ratio"], 0.08)
        self.assertEqual(by_name["h2o_heavy15_recent15"].metadata["h2o_heavy_ratio"], 0.15)
        self.assertEqual(by_name["h2o_heavy15_recent15"].metadata["h2o_recent_ratio"], 0.15)

    def test_config_runtime_uses_pilot_model_only(self) -> None:
        runtime = config_runtime({"model": {"pilot_model": "/tmp/pilot"}})
        self.assertEqual(runtime["pilot_model"], "/tmp/pilot")
        self.assertNotIn("profile_smoke_model", runtime)

    def test_exact_profiles_defaults_to_full_gpu_only(self) -> None:
        self.assertEqual(exact_profiles(["full_gpu", "kivi_4bit"]), {"full_gpu"})

    def test_conformal_guard_default_exact_profiles_is_full_gpu_only(self) -> None:
        guard = ConformalGuard(epsilon=0.2, delta=0.05, profiles=["full_gpu", "kivi_4bit"], calibration_rows=[])
        self.assertEqual(guard.exact_profiles, {"full_gpu"})

    def test_pilot_configs_use_full_gpu_only_formal_grid(self) -> None:
        pilot = load_config(Path("configs/pilot.yaml"))
        pilot_50 = load_config(Path("configs/pilot_50.yaml"))
        pilot_sharegpt = load_config(Path("configs/pilot_sharegpt.yaml"))
        pilot_phase1 = load_config(Path("configs/pilot_phase1.yaml"))
        pilot_session_trace = load_config(Path("configs/pilot_session_trace.yaml"))
        self.assertEqual(config_adapters(pilot), ["full", "kivi", "h2o"])
        self.assertEqual(config_profiles(pilot), FORMAL_PROFILES)
        self.assertEqual(config_profiles(pilot_50), FORMAL_PROFILES)
        self.assertEqual(config_profiles(pilot_sharegpt), FORMAL_PROFILES)
        self.assertEqual(config_profiles(pilot_phase1), FORMAL_PROFILES)
        self.assertEqual(config_profiles(pilot_session_trace), SESSION_TRACE_PROFILES)
        self.assertEqual(pilot["data"]["max_requests"], 200)
        self.assertEqual(pilot_50["data"]["max_requests"], 50)
        self.assertEqual(pilot["data"]["requests"], "data/fixtures/pilot_qa_summary_requests.jsonl")
        self.assertEqual(pilot["data"]["quality_mode"], "baseline")
        self.assertEqual(pilot_sharegpt["data"]["requests"], "data/fixtures/sharegpt_sessions.json")
        self.assertEqual(pilot_sharegpt["data"]["quality_mode"], "session_diagnostic")
        self.assertEqual(pilot["pilot"]["memory_budgets_mib"], [4900, 5000])
        self.assertEqual(pilot_50["pilot"]["memory_budgets_mib"], [4900, 5000])
        self.assertEqual(pilot_phase1["data"]["requests"], "data/fixtures/pilot_qa_summary_requests.jsonl")
        self.assertEqual(pilot_phase1["pilot"]["memory_budgets_mib"], [18, 26, 40])
        self.assertEqual(pilot_session_trace["data"]["requests"], "data/fixtures/pilot_session_trace_requests.jsonl")
        self.assertEqual(pilot_session_trace["data"]["quality_mode"], "session_diagnostic")
        self.assertEqual(pilot_session_trace["pilot"]["memory_budgets_mib"], [18, 26, 40])
        self.assertNotIn("profile_smoke_model", pilot["model"])
        self.assertNotIn("profile_smoke_model", pilot_50["model"])
        self.assertTrue(pilot["policies"]["record_rejected_unsafe"])
        self.assertTrue(pilot_50["policies"]["record_rejected_unsafe"])
        self.assertTrue(pilot_50["profile_smoke"]["require_ttft"])
        self.assertEqual(
            config_policies(pilot),
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
        )
        self.assertEqual(config_policies(pilot_50), ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"])
        self.assertEqual(exact_profiles(expected_profiles, pilot), {"full_gpu"})

    def test_registry_rejects_structured_static_policy_config_after_cleanup(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "未知 policy: static"):
            build_policies(
                [{"type": "static", "profile": "full_gpu", "name": "static_profile:full_gpu"}],
                rows,
                rows,
                ["full_gpu"],
                0.2,
                0.05,
                {"full_gpu"},
            )

    def test_measured_replay_pandas_mode_preserves_lookup_semantics(self) -> None:
        row = _measurement("r1", "full_gpu", 0.0)
        backend = MeasuredReplayBackend([row], use_pandas=True)
        result = backend.run([Request("r1", "qa", "p")], ["full_gpu"])[0]

        self.assertIsInstance(result, BackendResult)
        self.assertEqual(result.request_id, row.request_id)
        self.assertEqual(result.profile, row.profile)
        self.assertEqual(result.ttft_ms, row.ttft_ms)

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

    def test_csv_roundtrip_preserves_kv_cache_memory_mib(self) -> None:
        row = _measurement("r1", "full_gpu", 0.0, peak_memory_mib=20.0, kv_cache_memory_mib=7.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.csv"
            write_csv(path, [row.to_row()])
            [loaded] = __import__("run_util.io_utils", fromlist=["read_measurements"]).read_measurements(path)

        self.assertEqual(loaded.kv_cache_memory_mib, 7.5)

    def test_profile_summary_reports_kv_cache_memory_fields(self) -> None:
        rows = [
            _measurement("r1", "full_gpu", 0.0, kv_cache_memory_mib=10.0),
            _measurement("r2", "full_gpu", 0.0, kv_cache_memory_mib=20.0),
        ]
        summary = MetricCollector().summarize_profiles(rows)["full_gpu"]
        self.assertIn("mean_kv_cache_memory_mib", summary)
        self.assertIn("p95_kv_cache_memory_mib", summary)
        self.assertEqual(summary["mean_kv_cache_memory_mib"], 15.0)

    def test_metric_collector_profile_summary_uses_configured_epsilons(self) -> None:
        rows = [
            _measurement("r1", "full_gpu", 0.0),
            _measurement("r2", "full_gpu", 0.08),
            _measurement("r3", "full_gpu", 0.2),
        ]

        summary = MetricCollector().summarize_profiles(rows, epsilons=[0.05, 0.10])["full_gpu"]

        self.assertNotIn("violation_rate", summary)
        self.assertEqual(summary["violation_rate_eps0p05"], 2.0 / 3.0)
        self.assertEqual(summary["violation_rate_eps0p1"], 1.0 / 3.0)

    def test_profile_summary_rows_exclude_policy_columns_and_report_profile_metrics(self) -> None:
        rows = [
            ProfileMeasurement(
                request_id="r1",
                profile="full_gpu",
                adapter="full",
                ok=True,
                measured=True,
                latency_ms=10.0,
                ttft_ms=2.0,
                peak_memory_mib=100.0,
                resident_memory_mib=80.0,
                kv_cache_memory_mib=20.0,
                quality_loss=0.0,
                extra={
                    "task": "qa",
                    "length_bucket": "short",
                    "split": "calibration",
                    "family": "exact",
                    "backend": "transformers",
                    "ttft_semantics": "first_token",
                    "worker_mode": "batch",
                    "primary_metric": "em",
                },
            ),
            ProfileMeasurement(
                request_id="r2",
                profile="full_gpu",
                adapter="full",
                ok=True,
                measured=True,
                latency_ms=30.0,
                ttft_ms=6.0,
                peak_memory_mib=200.0,
                resident_memory_mib=180.0,
                kv_cache_memory_mib=40.0,
                quality_loss=0.2,
                extra={
                    "task": "summary",
                    "length_bucket": "long",
                    "split": "eval",
                    "family": "exact",
                    "backend": "transformers",
                    "ttft_semantics": "first_token",
                    "worker_mode": "batch",
                    "primary_metric": "rouge_l",
                },
            ),
        ]

        summary = profile_summary_rows(
            {"ok": True, "config": "pilot.yaml", "rows": {"profiles": 2}, "dry_run": False},
            rows,
            {
                "profiles": {"names": ["full_gpu"], "specs": {"full_gpu": {"exact": True}}},
                "pilot": {"epsilons": [0.05, 0.10], "deltas": [0.05]},
            },
        )

        profile_row = next(row for row in summary if row["section"] == "profile")
        for policy_column in (
            "policy_rows",
            "epsilon",
            "delta",
            "memory_budget_mib",
            "delta_slack",
            "worst_group_violation",
            "safe_ratio",
            "fallback_ratio",
            "exact_fallback_ratio",
            "exact_action_ratio",
            "lossy_action_ratio",
            "unique_action_count",
            "identical_to_full_lru",
            "unsafe_action_count",
            "candidate_safe_count",
            "action_distribution",
        ):
            self.assertNotIn(policy_column, profile_row)
        self.assertEqual(profile_row["family"], "exact")
        self.assertTrue(profile_row["exact"])
        self.assertEqual(profile_row["count"], 2.0)
        self.assertEqual(profile_row["p50_quality_loss"], 0.0)
        self.assertEqual(profile_row["p95_quality_loss"], 0.2)
        self.assertEqual(profile_row["p99_quality_loss"], 0.2)
        self.assertEqual(profile_row["violation_rate_eps0p05"], 0.5)
        self.assertEqual(profile_row["violation_rate_eps0p1"], 0.5)
        self.assertEqual(profile_row["p50_latency_ms"], 10.0)
        self.assertEqual(profile_row["p99_latency_ms"], 30.0)
        self.assertEqual(profile_row["p50_ttft_ms"], 2.0)
        self.assertEqual(profile_row["p99_peak_memory_mib"], 200.0)
        self.assertEqual(profile_row["p95_resident_memory_mib"], 180.0)
        self.assertEqual(profile_row["p99_kv_cache_memory_mib"], 40.0)
        self.assertEqual(profile_row["task_count"], 2.0)
        self.assertEqual(profile_row["length_bucket_count"], 2.0)
        self.assertEqual(profile_row["split_count"], 2.0)
        self.assertEqual(profile_row["backend_distribution"], {"transformers": 2})
        self.assertEqual(profile_row["ttft_semantics_distribution"], {"first_token": 2})
        self.assertEqual(profile_row["worker_mode_distribution"], {"batch": 2})
        self.assertEqual(profile_row["primary_metric_distribution"], {"em": 1, "rouge_l": 1})

    def test_validation_rejects_measured_rows_missing_kv_cache_memory(self) -> None:
        row = _measurement("r1", "full_gpu", 0.0)
        row = ProfileMeasurement(
            request_id=row.request_id,
            profile=row.profile,
            adapter=row.adapter,
            ok=row.ok,
            measured=row.measured,
            output_text=row.output_text,
            latency_ms=row.latency_ms,
            ttft_ms=row.ttft_ms,
            peak_memory_mib=row.peak_memory_mib,
            kv_cache_memory_mib=None,
            resident_memory_mib=row.resident_memory_mib,
            quality_loss=row.quality_loss,
            extra=row.extra,
        )
        with self.assertRaisesRegex(ValueError, "kv_cache_memory_mib"):
            validate_profile_measurements([row], require_measured=True)

    def test_policy_replay_copies_kv_cache_memory_mib(self) -> None:
        import run_util.run_policies as policy_runner
        from policies.full_lru import FullLRUPolicy

        request = Request("r1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        row = _measurement("r1", "full_gpu", 0.0, kv_cache_memory_mib=42.0)
        backend = MeasuredReplayBackend([row])

        [record] = policy_runner._run_policy_matrix([FullLRUPolicy()], [request], backend, {"full_gpu"})

        self.assertEqual(record.kv_cache_memory_mib, 42.0)
        self.assertEqual(record.backend_name, "measured_replay")

    def test_metrics_delta_slack_uses_delta_not_epsilon(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "kivi_4bit", True, True, quality_loss=0.3),
            PolicyRunRecord("p", "r2", "kivi_4bit", True, True, quality_loss=0.0),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["target_delta"], 0.05)
        self.assertEqual(summary["p"]["violation_rate"], 0.5)
        self.assertEqual(summary["p"]["delta_slack"], -0.45)

    def test_metrics_policy_summary_reports_p50_ttft_and_quality_loss(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "full_gpu", True, True, ttft_ms=10.0, quality_loss=0.0),
            PolicyRunRecord("p", "r2", "full_gpu", True, True, ttft_ms=20.0, quality_loss=0.1),
            PolicyRunRecord("p", "r3", "full_gpu", True, True, ttft_ms=30.0, quality_loss=0.2),
        ]

        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.15, delta=0.05, exact_profiles={"full_gpu"})["p"]

        self.assertEqual(summary["p50_ttft_ms"], 20.0)
        self.assertEqual(summary["p50_quality_loss"], 0.1)

    def test_metrics_reports_controller_oracle_and_aal_fields(self) -> None:
        records = [
            PolicyRunRecord(
                "p",
                "r1",
                "full_gpu",
                True,
                True,
                controller_overhead_ms=1.0,
                controller_qrp_ms=0.2,
                controller_cg_ms=0.3,
                controller_stc_ms=0.5,
                oracle_cost_ms=9.0,
                optimality_gap=0.1,
                audit_rate=0.05,
                drift_state="stable",
            )
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["controller_qrp_ms"], 0.2)
        self.assertEqual(summary["p"]["controller_cg_ms"], 0.3)
        self.assertEqual(summary["p"]["controller_stc_ms"], 0.5)
        self.assertEqual(summary["p"]["oracle_cost_ms"], 9.0)
        self.assertEqual(summary["p"]["optimality_gap"], 0.1)
        self.assertEqual(summary["p"]["audit_rate"], 0.05)
        self.assertEqual(summary["p"]["drift_state"], "stable")

    def test_policy_summary_reports_profile_optimization_diagnostics(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "full_gpu", True, True, rejected_profile="", candidate_safe_count=2.0),
            PolicyRunRecord("p", "r2", "full_gpu", True, True, rejected_profile="kivi_4bit_residual32", candidate_safe_count=1.0),
            PolicyRunRecord("q", "r1", "kivi_4bit_residual32", True, True, safe=True, candidate_safe_count=3.0),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["unique_action_count"], 1.0)
        self.assertTrue(summary["p"]["identical_to_full_lru"])
        self.assertEqual(summary["p"]["unsafe_action_count"], 1.0)
        self.assertEqual(summary["p"]["candidate_safe_count"], 1.5)
        self.assertFalse(summary["q"]["identical_to_full_lru"])

    def test_policy_summary_distinguishes_exact_fallback_ratio_from_exact_action_ratio(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "full_gpu", True, True, fallback_reason="forced fallback"),
            PolicyRunRecord("p", "r2", "full_gpu", True, True, fallback_reason=""),
            PolicyRunRecord("p", "r3", "kivi_4bit", True, True, fallback_reason=""),
        ]

        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})

        self.assertAlmostEqual(summary["p"]["exact_fallback_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["p"]["exact_action_ratio"], 2.0 / 3.0)

    def test_metrics_smoke_summary_fields_are_complete_and_nan_on_empty_values(self) -> None:
        records = [
            PolicyRunRecord(
                "p",
                "r1",
                "full_gpu",
                True,
                True,
                quality_loss=None,
                safe=None,
                controller_overhead_ms=None,
            )
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})["p"]
        for field in (
            "p95_ttft_ms",
            "p95_peak_memory_mib",
            "p95_kv_cache_memory_mib",
            "violation_rate",
            "safe_ratio",
            "fallback_ratio",
            "controller_overhead_ms",
        ):
            self.assertIn(field, summary)
        self.assertTrue(math.isnan(summary["p95_ttft_ms"]))
        self.assertTrue(math.isnan(summary["p95_peak_memory_mib"]))
        self.assertTrue(math.isnan(summary["violation_rate"]))
        self.assertTrue(math.isnan(summary["controller_overhead_ms"]))

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
        with self.assertRaisesRegex(ValueError, "profile 运行失败"):
            validate_profile_measurements([row], require_measured=True)

    def test_validate_profile_measurements_requires_measured_for_replay_failure(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="full_gpu",
            adapter="full",
            ok=True,
            measured=False,
            output_text="x",
            ttft_ms=None,
            peak_memory_mib=None,
            quality_loss=None,
            error="worker timeout",
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )
        with self.assertRaisesRegex(ValueError, "profile 运行失败"):
            validate_profile_measurements([row], require_measured=True)

    def test_validate_profile_measurements_rejects_empty_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile 表为空"):
            validate_profile_measurements([])

    def test_validate_profile_measurements_accepts_empty_measured_output_text(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="engine_full_lru",
            adapter="vllm_lru",
            ok=True,
            measured=True,
            output_text="",
            latency_ms=1.0,
            ttft_ms=1.0,
            peak_memory_mib=1.0,
            kv_cache_memory_mib=1.0,
            quality_loss=1.0,
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )
        validate_profile_measurements([row], require_measured=True, required_profiles=["engine_full_lru"])

    def test_validate_profile_measurements_rejects_lossy_without_full_baseline_quality(self) -> None:
        rows = [
            _measurement("r1", "kivi_4bit", None),
        ]
        with self.assertRaisesRegex(ValueError, "quality_loss"):
            validate_profile_measurements(rows, require_measured=True, required_profiles=["kivi_4bit"])

    def test_validate_profile_measurements_requires_first_token_ttft_when_configured(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="full_gpu",
            adapter="full",
            ok=True,
            measured=True,
            output_text="x",
            latency_ms=2.0,
            ttft_ms=None,
            peak_memory_mib=1.0,
            kv_cache_memory_mib=1.0,
            resident_memory_mib=1.0,
            quality_loss=0.0,
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )

        with self.assertRaisesRegex(ValueError, "ttft_ms"):
            validate_profile_measurements([row], require_ttft=True)
        validate_profile_measurements([row], require_ttft=False)

    def test_validate_profile_measurements_requires_first_token_semantics_when_configured(self) -> None:
        row = _measurement("r1", "full_gpu", 0.0)

        with self.assertRaisesRegex(ValueError, "ttft_semantics"):
            validate_profile_measurements([row], require_ttft=True)

        valid = ProfileMeasurement(
            request_id=row.request_id,
            profile=row.profile,
            adapter=row.adapter,
            ok=row.ok,
            measured=row.measured,
            output_text=row.output_text,
            latency_ms=row.latency_ms,
            ttft_ms=row.ttft_ms,
            peak_memory_mib=row.peak_memory_mib,
            kv_cache_memory_mib=row.kv_cache_memory_mib,
            resident_memory_mib=row.resident_memory_mib,
            quality_loss=row.quality_loss,
            extra={**row.extra, "ttft_semantics": "first_token"},
        )
        validate_profile_measurements([valid], require_ttft=True)

    def test_action_and_policy_record_expose_rejected_unsafe_fields(self) -> None:
        action = Action(
            profile="full_gpu",
            reason="fallback",
            rejected_profile="kivi_4bit_residual32",
            rejected_pred_loss=0.1,
            rejected_risk_upper=0.3,
            candidate_safe_count=1.0,
        )
        record = PolicyRunRecord(
            "utility_dynamic",
            "r1",
            action.profile,
            True,
            True,
            rejected_profile=action.rejected_profile,
            rejected_pred_loss=action.rejected_pred_loss,
            rejected_risk_upper=action.rejected_risk_upper,
            candidate_safe_count=action.candidate_safe_count,
        )
        row = record.to_row()
        self.assertEqual(row["rejected_profile"], "kivi_4bit_residual32")
        self.assertEqual(row["rejected_pred_loss"], 0.1)
        self.assertEqual(row["rejected_risk_upper"], 0.3)
        self.assertEqual(row["candidate_safe_count"], 1.0)

    def test_stats_policy_fastest_exact_uses_full_gpu_when_it_is_the_only_exact(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
        ]
        [policy] = build_policies(
            ["utility_dynamic"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(policy._fastest_exact_profile(), "full_gpu")

    def test_build_policies_accepts_record_rejected_unsafe_flag(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        [policy] = build_policies(
            ["utility_dynamic"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
            record_rejected_unsafe=True,
        )
        self.assertTrue(policy.record_rejected_unsafe)

    def test_e0_config_uses_pilot_model_only(self) -> None:
        config = load_config(Path("configs/e0_reproduce.yaml"))
        self.assertEqual(config_profiles(config), ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"])
        self.assertNotIn("profile_smoke_model", config["model"])
        self.assertEqual(config["data"]["requests"], "data/fixtures/e0_reproduce_requests.jsonl")

    def test_config_lists_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "profiles.adapters"):
            config_adapters({"profiles": {"names": ["full_gpu"]}, "policies": {"names": ["full_lru"]}})
        with self.assertRaisesRegex(ValueError, "profiles.names"):
            config_profiles({"profiles": {"adapters": ["full"]}, "policies": {"names": ["full_lru"]}})
        with self.assertRaisesRegex(ValueError, "policies.names"):
            config_policies({"profiles": {"adapters": ["full"], "names": ["full_gpu"]}, "policies": {}})

    def test_pilot_config_limits_profile_requests_to_200(self) -> None:
        config = load_config(Path("configs/pilot.yaml"))
        self.assertEqual(config_runtime(config)["max_requests"], 200)

    def test_limit_requests_by_split_balances_calibration_and_eval(self) -> None:
        requests = [
            Request(f"c{index}", "qa", "prompt", metadata={"split": "calibration"})
            for index in range(250)
        ] + [
            Request(f"e{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(250)
        ]

        limited = limit_requests_by_split(requests, 50)

        self.assertEqual(len(limited), 50)
        self.assertEqual(sum(request.metadata.get("split") == "calibration" for request in limited), 25)
        self.assertEqual(sum(request.metadata.get("split") == "eval" for request in limited), 25)

    def test_limit_requests_by_split_stratifies_task_and_length_bucket_within_each_split(self) -> None:
        requests = []
        for split, prefix in (("calibration", "c"), ("eval", "e")):
            for task, bucket in (
                ("qa_long_context", "long"),
                ("summary", "medium"),
                ("qa_long_context", "short"),
                ("summary", "long"),
            ):
                for index in range(8):
                    requests.append(
                        Request(
                            f"{prefix}_{task}_{bucket}_{index}",
                            task,
                            f"{task} {bucket} prompt {index}",
                            metadata={"split": split, "length_bucket": bucket},
                        )
                    )

        limited = limit_requests_by_split(requests, 16)

        self.assertEqual(len(limited), 16)
        self.assertEqual(sum(request.metadata.get("split") == "calibration" for request in limited), 8)
        self.assertEqual(sum(request.metadata.get("split") == "eval" for request in limited), 8)
        for split in ("calibration", "eval"):
            split_rows = [request for request in limited if request.metadata.get("split") == split]
            self.assertGreaterEqual(len({request.task for request in split_rows}), 2)
            self.assertGreaterEqual(len({request.metadata.get("length_bucket") for request in split_rows}), 2)
            self.assertGreaterEqual(
                len({(request.task, request.metadata.get("length_bucket")) for request in split_rows}),
                4,
            )

    def test_limit_requests_by_split_preserves_single_split_prefix(self) -> None:
        requests = [
            Request(f"e{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(23)
        ]

        limited = limit_requests_by_split(requests, 20)

        self.assertEqual([request.request_id for request in limited], [f"e{index}" for index in range(20)])

    def test_limit_requests_by_split_zero_keeps_all_requests(self) -> None:
        requests = [
            Request(f"r{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(3)
        ]

        self.assertIs(limit_requests_by_split(requests, 0), requests)

    def test_run_experiment_parser_only_exposes_pilot_smoke_measured(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured"])
        self.assertIs(args.func, pilot_smoke_measured)
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-policies"])

    def test_run_profile_test_parser_exposes_profile_only_arguments(self) -> None:
        parser = build_profile_test_parser()
        args = parser.parse_args(["--config", "configs/pilot_50.yaml", "--dry-run"])
        self.assertEqual(args.config, "configs/pilot_50.yaml")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.output, "")
        self.assertEqual(args.summary_output, "")
        with self.assertRaises(SystemExit):
            parser.parse_args(["pilot-smoke-measured"])

    def test_build_profile_table_uses_profile_many(self) -> None:
        class StubAdapter:
            name = "stub"

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                self.requests = list(requests)
                self.profile_name = profile_name
                self.dry_run = dry_run
                return [_measurement(request.request_id, profile_name, 0.0) for request in requests]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            output_path = Path(tmpdir) / "profiles.csv"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters:",
                        "    - stub",
                        "  names:",
                        "    - engine_full_lru",
                        "data:",
                        "  requests: data/fixtures/e0_reproduce_requests.jsonl",
                    ]
                ),
                encoding="utf-8",
            )
            stub = StubAdapter()
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}, "data": {"requests": "data/fixtures/e0_reproduce_requests.jsonl"}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
                patch(
                    "run_util.build_profile_table.load_requests",
                    return_value=(
                        [
                            Request("r1", "qa", "p", metadata={"split": "eval"}),
                            Request("r2", "qa", "q", metadata={"split": "eval"}),
                        ],
                        False,
                    ),
                ),
            ):
                code = build_profile_table(
                    argparse.Namespace(
                        config=str(config_path),
                        adapters=None,
                        output=str(output_path),
                        import_measurements="",
                        dry_run=False,
                    )
                )
            self.assertEqual(code, 0)
            self.assertFalse(stub.dry_run)
            self.assertEqual(stub.profile_name, "full_gpu")
            self.assertEqual([request.request_id for request in stub.requests], ["r1", "r2"])

    def test_build_profile_table_respects_max_requests_before_chunking(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = []

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                chunk = list(requests)
                self.calls.append([request.request_id for request in chunk])
                return [_measurement(request.request_id, profile_name, 0.0) for request in chunk]

        requests = [
            Request(f"c{index}", "qa", f"prompt {index}", metadata={"split": "calibration"})
            for index in range(25)
        ] + [
            Request(f"e{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(25)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1, "max_requests": 20}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
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

            self.assertEqual(code, 0)
            self.assertEqual([len(call) for call in stub.calls], [10, 10])
            self.assertEqual(stub.calls[0], [f"c{index}" for index in range(10)])
            self.assertEqual(stub.calls[1], [f"e{index}" for index in range(10)])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(sum(row["split"] == "calibration" for row in rows), 10)
            self.assertEqual(sum(row["split"] == "eval" for row in rows), 10)

    def test_build_profile_table_dry_run_covers_required_pilot_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pilot_profiles.csv"
            code = build_profile_table(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    adapters=None,
                    output=str(output_path),
                    import_measurements="",
                    dry_run=True,
                )
            )
            self.assertEqual(code, 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            profiles = {row["profile"] for row in rows}
            self.assertEqual(
                profiles,
                {
                    "full_gpu",
                    "kivi_4bit_residual32",
                    "kivi_4bit_residual64",
                    "kivi_2bit_residual32",
                    "kivi_2bit_residual64",
                    "h2o_heavy10_recent10",
                    "h2o_heavy15_recent15",
                    "h2o_heavy20_recent20",
                },
            )
            self.assertTrue(all(row["measured"] == "False" for row in rows))

    def test_run_policies_runs_five_baselines_on_pilot_dry_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"

            build_code = build_profile_table(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    adapters=None,
                    output=str(profile_path),
                    import_measurements="",
                    dry_run=True,
                )
            )
            self.assertEqual(build_code, 0)

            policy_code = run_policies(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    measurements=str(profile_path),
                    output=str(policy_path),
                    profiles=None,
                    policies=None,
                    policy_config=None,
                    epsilon=None,
                    delta=None,
                    memory_budget_mib=None,
                    use_pandas_replay=False,
                    allow_dry_run_replay=True,
                )
            )
            self.assertEqual(policy_code, 0)

            with policy_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["policy"] for row in rows},
                {"full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"},
            )
            self.assertIn("rejected_profile", rows[0])
            self.assertIn("candidate_safe_count", rows[0])

    def test_build_profile_table_chunks_profile_many_and_writes_incrementally(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = []

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                chunk = list(requests)
                self.calls.append([request.request_id for request in chunk])
                return [_measurement(request.request_id, profile_name, 0.0) for request in chunk]

        requests = [
            Request(f"r{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(23)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            append_counts = []
            append_impl = profile_table_module._append_profile_rows

            def record_append(path, rows):
                append_counts.append(len(rows))
                append_impl(path, rows)

            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_util.build_profile_table._append_profile_rows", side_effect=record_append),
            ):
                stream = io.StringIO()
                with redirect_stderr(stream):
                    code = build_profile_table(
                        argparse.Namespace(
                            config="config.yaml",
                            adapters=None,
                            output=str(output_path),
                            import_measurements="",
                            dry_run=False,
                        )
                    )

            self.assertEqual(code, 0)
            self.assertEqual([len(call) for call in stub.calls], [10, 10, 3])
            self.assertEqual(append_counts, [10, 10, 3])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 23)
            progress = [
                json.loads(line)
                for line in stream.getvalue().splitlines()
                if '"event": "profile_chunk_complete"' in line
            ]
            self.assertEqual([row["chunk_index"] for row in progress], [1, 2, 3])
            self.assertEqual([row["completed_requests"] for row in progress], [10, 20, 23])
            self.assertTrue(all(row["output"] == str(output_path) for row in progress))

    def test_build_profile_table_chunk_failure_keeps_completed_output(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = 0

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                self.calls += 1
                if self.calls == 2:
                    return [
                        ProfileMeasurement(
                            request_id=request.request_id,
                            profile=profile_name,
                            adapter="stub",
                            ok=False,
                            measured=False,
                            error="chunk failed",
                        )
                        for request in requests
                    ]
                return [_measurement(request.request_id, profile_name, 0.0) for request in requests]

        requests = [
            Request(f"r{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = build_profile_table(
                        argparse.Namespace(
                            config="config.yaml",
                            adapters=None,
                            output=str(output_path),
                            import_measurements="",
                            dry_run=False,
                        )
                    )

            self.assertEqual(code, 2)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["request_id"] for row in rows], [f"r{index}" for index in range(10)])
            diagnostic_path = Path(tmpdir) / "profiles_failed_chunks.csv"
            self.assertTrue(diagnostic_path.exists())
            with diagnostic_path.open("r", encoding="utf-8", newline="") as handle:
                failed_rows = list(csv.DictReader(handle))
            self.assertEqual([row["request_id"] for row in failed_rows], ["r10", "r11"])
            payload = json.loads(stream.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["diagnostic_output"], str(diagnostic_path))
            self.assertEqual(payload["failures"][0]["error"], "chunk failed")

    def test_build_profile_table_restarts_persistent_worker_when_cuda_binding_changes(self) -> None:
        class StubAdapter:
            name = "stub"

            def profiles(self):
                return (
                    ProfileSpec(
                        "full_gpu",
                        "full",
                        "env",
                        lossy=False,
                        exact=True,
                        metadata={
                            "device_strategy": "balanced_two_gpu",
                            "cuda_visible_devices": "0,1",
                        },
                    ),
                    ProfileSpec(
                        "kivi_4bit_residual64",
                        "kivi",
                        "env",
                        lossy=True,
                        exact=False,
                        metadata={
                            "device_strategy": "single_gpu",
                            "cuda_visible_devices": "2",
                        },
                    ),
                )

        class StubWorker:
            def __init__(self, worker_id: int) -> None:
                self.worker_id = worker_id
                self.closed = False

            def close(self) -> None:
                self.closed = True

        requests = [Request("r0", "qa", "prompt", metadata={"split": "eval"})]
        created_workers: list[StubWorker] = []
        worker_ids_seen: list[int] = []

        def fake_create_worker(adapter, runtime_config):
            worker = StubWorker(len(created_workers) + 1)
            created_workers.append(worker)
            return worker

        def fake_profile_many_compat(
            adapter,
            request_chunk,
            profile_name,
            *,
            dry_run,
            session_runtime,
            memory_budget_mib,
            persistent_worker,
        ):
            worker_ids_seen.append(getattr(persistent_worker, "worker_id", 0))
            return [_measurement(request.request_id, profile_name, 0.0) for request in request_chunk]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu", "kivi_4bit_residual64"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu", "kivi_4bit_residual64"]),
                patch(
                    "run_util.build_profile_table.config_runtime",
                    return_value={
                        "repeat": 1,
                        "use_persistent_workers": True,
                        "timeout_s": 180,
                        "full_device_strategy": "balanced_two_gpu",
                        "full_cuda_visible_devices": "0,1",
                        "kivi_device_strategy": "single_gpu",
                        "kivi_cuda_visible_devices": "2",
                    },
                ),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[StubAdapter()]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_util.build_profile_table.create_persistent_worker", side_effect=fake_create_worker),
                patch("run_util.build_profile_table._profile_many_compat", side_effect=fake_profile_many_compat),
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

        self.assertEqual(code, 0)
        self.assertEqual(worker_ids_seen, [1, 2])
        self.assertEqual(len(created_workers), 2)
        self.assertTrue(all(worker.closed for worker in created_workers))

    def test_build_profile_table_success_clears_stale_failed_chunks_output(self) -> None:
        class StubAdapter:
            name = "stub"

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                return [_measurement(request.request_id, profile_name, 0.0) for request in requests]

        requests = [
            Request(f"r{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            diagnostic_path = Path(tmpdir) / "profiles_failed_chunks.csv"
            diagnostic_path.write_text("adapter,error\nstub,old failure\n", encoding="utf-8")

            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[StubAdapter()]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
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

            self.assertEqual(code, 0)
            self.assertTrue(output_path.exists())
            self.assertFalse(diagnostic_path.exists())

    def test_pilot_smoke_measured_builds_profiles_without_dry_run_and_replays_same_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            profiles = FORMAL_PROFILES
            calls = {}

            def fake_build(args: argparse.Namespace) -> int:
                calls["profile_args"] = args
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in profiles])
                print(json.dumps({"output": args.output, "rows": len(profiles), "summary": {"profiles": len(profiles)}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                calls["policy_args"] = args
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 7,
                            "epsilon": 0.05,
                            "delta": 0.05,
                            "memory_budget_mib": 6144.0,
                            "summary": {"full_lru": {"p95_ttft_ms": 1.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_util.experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_util.experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_util.experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertEqual(stream.getvalue(), "")
            self.assertFalse(calls["profile_args"].dry_run)
            self.assertEqual(calls["profile_args"].output, str(profile_path))
            self.assertEqual(calls["policy_args"].measurements, str(profile_path))
            self.assertEqual(calls["policy_args"].output, str(policy_path))
            self.assertFalse(calls["policy_args"].allow_dry_run_replay)
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["section"], "experiment")
            self.assertEqual(summary_rows[0]["ok"], "True")
            self.assertEqual(summary_rows[0]["profile_rows"], "8")
            self.assertEqual(summary_rows[0]["policy_rows"], "7")
            full_lru = next(row for row in summary_rows if row["section"] == "policy" and row["name"] == "full_lru")
            self.assertEqual(full_lru["p95_ttft_ms"], "1.0")
            self.assertLess(
                list(summary_rows[0]).index("mean_ttft_ms"),
                list(summary_rows[0]).index("action_distribution"),
            )
            self.assertLess(
                list(summary_rows[0]).index("exact_fallback_ratio"),
                list(summary_rows[0]).index("exact_action_ratio"),
            )
            for deleted in ("return_code", "step", "profile_output", "policy_output", "summary_output"):
                self.assertNotIn(deleted, summary_rows[0])

    def test_pilot_test_config_includes_tailguard_and_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(
                config_path,
                Path(tmpdir) / "profiles.csv",
                Path(tmpdir) / "policy.csv",
                Path(tmpdir) / "summary.csv",
            )

            self.assertEqual(
                config_policies(load_config(config_path)),
                ["full_lru", "static_best", "static_safe", "tailguard", "quality_oracle", "utility_dynamic", "uncalibrated_dynamic"],
            )

    def test_policy_sweep_points_expand_cartesian_product(self) -> None:
        config = {
            "pilot": {
                "epsilons": [0.05, 0.10],
                "deltas": [0.05],
                "memory_budgets_mib": [4900, 5000],
            }
        }

        self.assertEqual(
            _policy_sweep_points(config),
            [
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 4900.0},
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 5000.0},
                {"epsilon": 0.10, "delta": 0.05, "memory_budget_mib": 4900.0},
                {"epsilon": 0.10, "delta": 0.05, "memory_budget_mib": 5000.0},
            ],
        )

    def test_policy_output_for_sweep_keeps_single_run_path(self) -> None:
        self.assertEqual(
            _policy_output_for_sweep(
                "out/policy_tables/pilot_policy.csv",
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 4900.0},
                1,
            ),
            "out/policy_tables/pilot_policy.csv",
        )

    def test_policy_output_for_sweep_namespaces_multi_run_path(self) -> None:
        self.assertEqual(
            _policy_output_for_sweep(
                "out/policy_tables/pilot_policy.csv",
                {"epsilon": 0.05, "delta": 0.10, "memory_budget_mib": 5000.0},
                4,
            ),
            "out/policy_tables/pilot_policy_eps0p05_delta0p1_mem5000.csv",
        )

    def test_pilot_smoke_measured_runs_all_policy_sweep_outputs_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.05, 0.10]",
                        "  deltas: [0.05]",
                        "  memory_budgets_mib: [4900, 5000]",
                        "outputs:",
                        f"  smoke_profiles: {profile_path}",
                        f"  smoke_policy: {policy_path}",
                        f"  smoke_summary: {summary_path}",
                    ]
                ),
                encoding="utf-8",
            )
            policy_calls = []

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                policy_calls.append((args.epsilon, args.delta, args.memory_budget_mib, args.output))
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 2,
                            "epsilon": float(args.epsilon),
                            "delta": float(args.delta),
                            "memory_budget_mib": float(args.memory_budget_mib),
                            "summary": {"full_lru": {"count": 2.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertEqual(len(policy_calls), 4)
            self.assertEqual(policy_calls[0][0:3], (0.05, 0.05, 4900.0))
            self.assertEqual(policy_calls[-1][0:3], (0.10, 0.05, 5000.0))
            self.assertEqual(len({call[3] for call in policy_calls}), 4)
            self.assertTrue(all(call[3] != str(policy_path) for call in policy_calls))

    def test_pilot_summary_contains_each_policy_sweep_row(self) -> None:
        rows = _summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "rows": {"profiles": 10, "policy": 4},
                "profile": {"summary": {}},
                "policy_runs": [
                    {
                        "ok": True,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 4900.0,
                        "payload": {"summary": {"full_lru": {"count": 2.0, "p95_ttft_ms": 1.0}}},
                    },
                    {
                        "ok": True,
                        "epsilon": 0.1,
                        "delta": 0.05,
                        "memory_budget_mib": 4900.0,
                        "payload": {"summary": {"full_lru": {"count": 2.0, "p95_ttft_ms": 2.0}}},
                    },
                ],
            }
        )

        policy_rows = [row for row in rows if row["section"] == "policy" and row["name"] == "full_lru"]
        self.assertEqual(len(policy_rows), 2)
        self.assertEqual([row["epsilon"] for row in policy_rows], [0.05, 0.1])
        self.assertEqual([row["p95_ttft_ms"] for row in policy_rows], [1.0, 2.0])

    def test_total_policy_summary_rows_flatten_policy_sweeps_only(self) -> None:
        rows = total_policy_summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "run_dir": "out/manual",
                "profile": {"summary": {"full_gpu": {"count": 3.0}}},
                "policy_runs": [
                    {
                        "ok": True,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 4900.0,
                        "payload": {
                            "summary": {
                                "full_lru": {
                                    "mean_ttft_ms": 100.0,
                                    "p50_ttft_ms": 90.0,
                                    "p95_ttft_ms": 120.0,
                                    "mean_quality_loss": 0.0,
                                    "p50_quality_loss": 0.0,
                                    "p95_quality_loss": 0.0,
                                    "violation_rate": 0.0,
                                    "worst_group_violation": 0.0,
                                    "mean_kv_cache_memory_mib": 4000.0,
                                    "p95_kv_cache_memory_mib": 4100.0,
                                    "controller_overhead_ms": 0.2,
                                    "action_distribution": {"full_gpu": 3},
                                },
                                "tailguard": {
                                    "mean_ttft_ms": 80.0,
                                    "p50_ttft_ms": 70.0,
                                    "p95_ttft_ms": 95.0,
                                    "mean_quality_loss": 0.02,
                                    "p50_quality_loss": 0.01,
                                    "p95_quality_loss": 0.04,
                                    "violation_rate": 0.0,
                                    "worst_group_violation": 0.0,
                                    "mean_kv_cache_memory_mib": 3000.0,
                                    "p95_kv_cache_memory_mib": 3100.0,
                                    "controller_overhead_ms": 0.4,
                                    "action_distribution": {"kivi": 2, "full_gpu": 1},
                                },
                            }
                        },
                    },
                    {
                        "ok": True,
                        "epsilon": 0.1,
                        "delta": 0.05,
                        "memory_budget_mib": 5000.0,
                        "payload": {"summary": {"full_lru": {"p95_ttft_ms": 110.0}, "tailguard": {"p95_ttft_ms": 85.0}}},
                    },
                ],
            }
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["policy"] for row in rows}, {"full_lru", "tailguard"})
        self.assertNotIn("section", rows[0])
        self.assertNotIn("name", rows[0])
        first = rows[0]
        self.assertEqual(first["config"], "pilot.yaml")
        self.assertEqual(first["run_dir"], "out/manual")
        self.assertEqual(first["policy"], "full_lru")
        self.assertEqual(first["memory_budget_mib"], 4900.0)
        self.assertEqual(first["epsilon"], 0.05)
        self.assertEqual(first["delta"], 0.05)
        for field in (
            "mean_ttft_ms",
            "p50_ttft_ms",
            "p95_ttft_ms",
            "mean_quality_loss",
            "p50_quality_loss",
            "p95_quality_loss",
            "violation_rate",
            "worst_group_violation",
            "mean_kv_cache_memory_mib",
            "p95_kv_cache_memory_mib",
            "controller_overhead_ms",
            "action_distribution",
        ):
            self.assertIn(field, first)

    def test_summary_rows_keep_profile_and_policy_metrics_in_separate_sections(self) -> None:
        rows = summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "epsilon": 0.05,
                "delta": 0.05,
                "memory_budget_mib": 4900.0,
                "profile": {
                    "summary": {
                        "full_gpu": {
                            "count": 2.0,
                            "measured_count": 2.0,
                            "mean_ttft_ms": 9.0,
                        }
                    }
                },
                "policy": {
                    "summary": {
                        "tailguard": {
                            "count": 1.0,
                            "mean_ttft_ms": 7.0,
                            "mean_kv_cache_memory_mib": 44.0,
                            "action_distribution": {"kivi": 1},
                        }
                    }
                },
            }
        )

        profile_row = next(row for row in rows if row["section"] == "profile")
        policy_row = next(row for row in rows if row["section"] == "policy")

        self.assertEqual(profile_row["name"], "full_gpu")
        self.assertEqual(profile_row["mean_ttft_ms"], 9.0)
        self.assertNotIn("action_distribution", profile_row)
        self.assertEqual(policy_row["name"], "tailguard")
        self.assertEqual(policy_row["mean_ttft_ms"], 7.0)
        self.assertEqual(policy_row["mean_kv_cache_memory_mib"], 44.0)
        self.assertEqual(policy_row["action_distribution"], {"kivi": 1})

    def test_backend_result_from_profile_measurement_marks_replay_source(self) -> None:
        measurement = _measurement("r1", "full_gpu", 0.0, kv_cache_memory_mib=12.0)

        result = BackendResult.from_profile_measurement(measurement, backend_name="measured_replay")

        self.assertEqual(result.profile, "full_gpu")
        self.assertEqual(result.kv_cache_memory_mib, 12.0)
        self.assertEqual(result.backend_name, "measured_replay")
        self.assertEqual(result.replay_source, "measured_profile_table")

    def test_run_policy_matrix_rejects_invalid_backend_result(self) -> None:
        import run_util.run_policies as policy_runner
        from policies.full_lru import FullLRUPolicy

        class InvalidBackend:
            def __init__(self) -> None:
                self.cache_state = policy_runner.CacheState()

            def run(self, requests, profiles):
                return [
                    BackendResult(
                        request_id=requests[0].request_id,
                        session_id=requests[0].session_id,
                        turn_index=requests[0].turn_index,
                        profile=profiles[0],
                        ok=True,
                        measured=True,
                        latency_ms=10.0,
                        peak_memory_mib=20.0,
                    )
                ]

        request = Request("r1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})

        [record] = policy_runner._run_policy_matrix([FullLRUPolicy()], [request], InvalidBackend(), {"full_gpu"})

        self.assertFalse(record.ok)
        self.assertIn("kv_cache_memory_mib", record.error or "")

    def test_summary_reports_h0_h1_h2_lite_evidence_fields(self) -> None:
        rows = _summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "profiles": FORMAL_PROFILES,
                "policies": ["full_lru", "static_best", "static_safe", "tailguard", "quality_oracle", "utility_dynamic", "uncalibrated_dynamic"],
                "rows": {"profiles": 8, "policy": 7},
                "profile": {"summary": {"full_gpu": {"p95_ttft_ms": 1.0, "p99_quality_loss": 0.0, "cvar_quality_loss": 0.0}}},
                "policy_runs": [
                    {
                        "ok": True,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 4900.0,
                        "payload": {
                            "summary": {
                                "full_lru": {
                                    "p95_ttft_ms": 1.0,
                                    "p95_quality_loss": 0.0,
                                    "mean_kv_cache_memory_mib": 100.0,
                                    "fallback_ratio": 0.0,
                                    "oracle": False,
                                },
                                "tailguard": {
                                    "p95_ttft_ms": 0.8,
                                    "worst_group_violation": 0.0,
                                    "mean_kv_cache_memory_mib": 80.0,
                                    "fallback_ratio": 0.1,
                                    "oracle": False,
                                },
                                "quality_oracle": {"oracle": True},
                            }
                        },
                    }
                ],
            }
        )

        experiment = rows[0]
        self.assertTrue(experiment["has_h0_tail_metrics"])
        self.assertTrue(experiment["has_h1_coverage_metrics"])
        self.assertTrue(experiment["has_h2_lite_benefit_metrics"])
        self.assertEqual(experiment["deployable_baseline_names"], ["full_lru", "static_best", "static_safe", "tailguard", "utility_dynamic", "uncalibrated_dynamic"])

    def test_summary_evidence_prefers_policy_metrics_when_policy_section_exists(self) -> None:
        rows = _summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "policies": ["full_lru", "tailguard"],
                "profile": {"summary": {"full_gpu": {"p95_quality_loss": 0.2}}},
                "policy": {"summary": {"tailguard": {"mean_ttft_ms": 8.0}}},
            }
        )

        experiment = rows[0]

        self.assertFalse(experiment["has_h0_tail_metrics"])
        self.assertFalse(experiment["has_h1_coverage_metrics"])
        self.assertFalse(experiment["has_h2_lite_benefit_metrics"])

    def test_pilot_summary_experiment_row_includes_run_dir_and_visual_outputs(self) -> None:
        rows = _summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "run_dir": "out/manual",
                "visual_outputs": ["out/manual/policy_tables/summary_policy_p95_ttft.png"],
                "rows": {"profiles": 1, "policy": 1},
            }
        )

        self.assertEqual(rows[0]["run_dir"], "out/manual")
        self.assertEqual(rows[0]["visual_outputs"], ["out/manual/policy_tables/summary_policy_p95_ttft.png"])

    def test_pilot_summary_experiment_error_uses_failed_policy_sweep_payload(self) -> None:
        rows = _summary_rows(
            {
                "ok": False,
                "config": "pilot.yaml",
                "rows": {"profiles": 10, "policy": 2},
                "policy_runs": [
                    {
                        "ok": False,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 4900.0,
                        "payload": {"error": "policy failed"},
                    },
                ],
            }
        )

        self.assertEqual(rows[0]["error"], "policy failed")

    def test_run_experiment_pilot_accepts_config_argument(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured", "--config", "configs/pilot_50.yaml"])
        self.assertEqual(args.config, "configs/pilot_50.yaml")

    def test_run_experiment_pilot_accepts_run_dir_argument(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured", "--config", "configs/pilot_50.yaml", "--run-dir", "out/manual"])

        self.assertEqual(args.run_dir, "out/manual")

    def test_run_experiment_pilot_accepts_total_summary_output_argument(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured", "--config", "configs/pilot_50.yaml", "--total-summary-output", "custom.csv"])

        self.assertEqual(args.total_summary_output, "custom.csv")

    def test_default_run_dir_uses_timestamp_and_config_stem(self) -> None:
        fixed_time = SimpleNamespace(strftime=lambda fmt: "20260802_123456")

        with patch("run_util.experiment.datetime") as datetime_mock:
            datetime_mock.now.return_value = fixed_time
            run_dir = run_experiment._resolve_run_dir(None, "configs/pilot_50.yaml")

        self.assertEqual(run_dir, Path("out/20260802_123456_pilot_50"))

    def test_explicit_run_dir_is_honored(self) -> None:
        self.assertEqual(
            run_experiment._resolve_run_dir("out/manual_pilot_50_visual", "configs/pilot_50.yaml"),
            Path("out/manual_pilot_50_visual"),
        )

    def test_relative_output_paths_are_relocated_under_run_dir(self) -> None:
        run_dir = Path("out/manual")

        self.assertEqual(
            run_experiment._resolve_run_output("out/policy_tables/pilot_50_summary.csv", run_dir),
            Path("out/manual/policy_tables/pilot_50_summary.csv"),
        )

    def test_absolute_output_paths_are_not_relocated(self) -> None:
        output = Path(tempfile.gettempdir()) / "tailguardkv_summary.csv"

        self.assertEqual(run_experiment._resolve_run_output(str(output), Path("out/manual")), output)

    def test_pilot_smoke_measured_writes_configured_summary_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "configured_profiles.csv"
            policy_path = Path(tmpdir) / "configured_policy.csv"
            summary_path = Path(tmpdir) / "configured_summary.csv"
            config_path = Path(tmpdir) / "pilot_custom.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.2]",
                        "  deltas: [0.05]",
                        "outputs:",
                        f"  smoke_profiles: {profile_path}",
                        f"  smoke_policy: {policy_path}",
                        f"  smoke_summary: {summary_path}",
                    ]
                ),
                encoding="utf-8",
            )

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(json.dumps({"output": args.output, "rows": 1, "epsilon": 0.2, "delta": 0.05, "summary": {"full_lru": {"count": 1.0}}}))
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertTrue(summary_path.exists())

    def test_pilot_smoke_measured_writes_explicit_total_summary_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            total_summary_path = Path(tmpdir) / "custom_total.csv"
            config_path = Path(tmpdir) / "pilot_custom.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in FORMAL_PROFILES])
                print(json.dumps({"output": args.output, "rows": len(FORMAL_PROFILES), "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 1,
                            "epsilon": args.epsilon,
                            "delta": args.delta,
                            "memory_budget_mib": args.memory_budget_mib,
                            "summary": {"full_lru": {"p95_ttft_ms": 1.0, "violation_rate": 0.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", return_value=[]),
            ):
                code = pilot_smoke_measured(
                    argparse.Namespace(config=str(config_path), run_dir=None, total_summary_output=str(total_summary_path))
                )

            self.assertEqual(code, 0)
            with total_summary_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["policy"], "full_lru")
            self.assertEqual(rows[0]["p95_ttft_ms"], "1.0")
            self.assertEqual(rows[0]["violation_rate"], "0.0")

    def test_pilot_smoke_measured_derives_default_total_summary_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "pilot_50_measured_summary.csv"
            total_summary_path = Path(tmpdir) / "pilot_50_measured_total_summary.csv"
            config_path = Path(tmpdir) / "pilot_custom.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in FORMAL_PROFILES])
                print(json.dumps({"output": args.output, "rows": len(FORMAL_PROFILES), "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_lru": {"p95_ttft_ms": 1.0}}}))
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", return_value=[]),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=None))

            self.assertEqual(code, 0)
            self.assertTrue(total_summary_path.exists())

    def test_pilot_smoke_measured_relocates_relative_outputs_under_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "manual_run"
            config_path = root / "pilot_custom.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.2]",
                        "  deltas: [0.05]",
                        "outputs:",
                        "  smoke_profiles: out/profile_tables/configured_profiles.csv",
                        "  smoke_policy: out/policy_tables/configured_policy.csv",
                        "  smoke_summary: out/policy_tables/configured_summary.csv",
                    ]
                ),
                encoding="utf-8",
            )
            calls = {}

            def fake_build(args: argparse.Namespace) -> int:
                calls["profile_output"] = args.output
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                calls["policy_measurements"] = args.measurements
                calls["policy_output"] = args.output
                print(json.dumps({"output": args.output, "rows": 1, "epsilon": 0.2, "delta": 0.05, "summary": {"full_lru": {"count": 1.0}}}))
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", return_value=[]),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=str(run_dir)))

            self.assertEqual(code, 0)
            self.assertEqual(calls["profile_output"], str(run_dir / "profile_tables/configured_profiles.csv"))
            self.assertEqual(calls["policy_measurements"], str(run_dir / "profile_tables/configured_profiles.csv"))
            self.assertEqual(calls["policy_output"], str(run_dir / "policy_tables/configured_policy.csv"))
            self.assertTrue((run_dir / "policy_tables/configured_summary.csv").exists())

    def test_pilot_smoke_measured_relocates_configured_total_summary_under_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "manual_run"
            config_path = root / "pilot_custom.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.2]",
                        "  deltas: [0.05]",
                        "outputs:",
                        "  smoke_profiles: out/profile_tables/profiles.csv",
                        "  smoke_policy: out/policy_tables/policy.csv",
                        "  smoke_summary: out/policy_tables/summary.csv",
                        "  smoke_total_summary: out/policy_tables/configured_total.csv",
                    ]
                ),
                encoding="utf-8",
            )
            payloads = []
            plot_inputs = []

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_lru": {"p95_ttft_ms": 1.0}}}))
                return 0

            def record_payload(payload: dict[str, object]) -> None:
                payloads.append(payload)
                run_experiment.write_summary(payload, str(payload.get("summary_output")))
                run_experiment.write_total_policy_summary(payload, str(payload.get("total_summary_output")))

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", return_value=[]),
                patch("run_util.experiment._print_and_write", side_effect=record_payload),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=str(run_dir)))

            self.assertEqual(code, 0)
            self.assertTrue((run_dir / "policy_tables/configured_total.csv").exists())
            self.assertEqual(payloads[-1]["total_summary_output"], str(run_dir / "policy_tables/configured_total.csv"))

    def test_policy_sweep_suffixes_apply_inside_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "manual_run"
            config_path = root / "pilot_custom.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.05, 0.10]",
                        "  deltas: [0.05]",
                        "  memory_budgets_mib: [4900]",
                        "outputs:",
                        "  smoke_profiles: out/profile_tables/profiles.csv",
                        "  smoke_policy: out/policy_tables/policy.csv",
                        "  smoke_summary: out/policy_tables/summary.csv",
                    ]
                ),
                encoding="utf-8",
            )
            policy_outputs = []

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                policy_outputs.append(args.output)
                print(json.dumps({"output": args.output, "rows": 1, "epsilon": args.epsilon, "delta": args.delta, "summary": {"full_lru": {"count": 1.0}}}))
                return 0

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", return_value=[]),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=str(run_dir)))

            self.assertEqual(code, 0)
            self.assertEqual(
                policy_outputs,
                [
                    str(run_dir / "policy_tables/policy_eps0p05_delta0p05_mem4900.csv"),
                    str(run_dir / "policy_tables/policy_eps0p1_delta0p05_mem4900.csv"),
                ],
            )

    def test_pilot_smoke_measured_records_generated_visual_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            chart_path = Path(tmpdir) / "summary_policy_p95_ttft.png"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            payloads = []
            plot_inputs = []

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in FORMAL_PROFILES])
                print(json.dumps({"output": args.output, "rows": len(FORMAL_PROFILES), "summary": {"profiles": len(FORMAL_PROFILES)}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(json.dumps({"output": args.output, "rows": 1, "epsilon": 0.05, "delta": 0.05, "summary": {"full_lru": {"p95_ttft_ms": 1.0}}}))
                return 0

            def record_payload(payload: dict[str, object]) -> None:
                payloads.append(payload)

            with (
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies", side_effect=fake_policies),
                patch("run_util.experiment.plot_summary", side_effect=lambda path: plot_inputs.append(path) or [chart_path]),
                patch("run_util.experiment._print_and_write", side_effect=record_payload),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path), run_dir=None))

            self.assertEqual(code, 0)
            self.assertEqual(payloads[-1]["visual_outputs"], [str(chart_path)])
            self.assertEqual(plot_inputs, [str(summary_path.with_name("summary_total_summary.csv"))])

    def test_visual_plot_summary_creates_policy_line_charts_by_budget_constraint_cell(self) -> None:
        from visual.plot_summary import plot_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "total_summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,p95_ttft_ms,mean_kv_cache_memory_mib,p95_quality_loss,violation_rate",
                        "full_lru,4900,0.05,0.05,120,800,0.01,0.0",
                        "tailguard,4900,0.05,0.05,90,600,0.03,0.02",
                        "full_lru,5000,0.05,0.05,130,850,0.01,0.0",
                        "tailguard,5000,0.05,0.05,95,610,0.02,0.01",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = plot_summary(summary_path)

            self.assertEqual(
                {path.name for path in outputs},
                {
                    "summary_policy_p95_ttft.png",
                    "summary_policy_kv_memory.png",
                    "summary_policy_quality_loss.png",
                    "summary_policy_violation_rate.png",
                },
            )
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))

    def test_visual_plot_summary_skips_missing_numeric_data_without_raising(self) -> None:
        from visual.plot_summary import plot_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "policy,memory_budget_mib,epsilon,delta,p95_ttft_ms",
                        "full_lru,4900,0.05,0.05,120",
                        "tailguard,4900,0.05,0.05,90",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = plot_summary(summary_path)

            self.assertEqual({path.name for path in outputs}, {"summary_policy_p95_ttft.png"})

    def test_pilot_smoke_measured_stops_when_profile_stage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)

            def fake_build(args: argparse.Namespace) -> int:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "output": args.output,
                            "diagnostic_output": str(Path(tmpdir) / "profiles_failed_chunks.csv"),
                            "error": "profile failed",
                            "failures": [{"request_id": "r10", "error": "chunk failed"}],
                        }
                    )
                )
                return 2

            with (
                patch("run_util.experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_util.experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_util.experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies") as policy_mock,
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["ok"], "False")
            self.assertEqual(summary_rows[0]["error"], "profile failed")
            self.assertEqual(summary_rows[0]["diagnostic_output"], str(Path(tmpdir) / "profiles_failed_chunks.csv"))
            self.assertEqual(json.loads(summary_rows[0]["failures"])[0]["error"], "chunk failed")
            self.assertNotIn("step", summary_rows[0])
            policy_mock.assert_not_called()

    def test_pilot_smoke_measured_rejects_unmeasured_profile_before_policy_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            profiles = FORMAL_PROFILES

            def fake_build(args: argparse.Namespace) -> int:
                rows = [_measurement("e1", profile, 0.0, measured=(profile != "kivi_4bit_residual32")).to_row() for profile in profiles]
                write_csv(Path(args.output), rows)
                print(json.dumps({"output": args.output, "rows": len(rows)}))
                return 0

            with (
                patch("run_util.experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_util.experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_util.experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_util.experiment.build_profile_table", side_effect=fake_build),
                patch("run_util.experiment.run_policies") as policy_mock,
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["ok"], "False")
            self.assertIn("profile 运行失败", summary_rows[0]["error"])
            self.assertIn("measured=False", summary_rows[0]["error"])
            self.assertNotIn("step", summary_rows[0])
            policy_mock.assert_not_called()

    def test_run_profile_test_writes_experiment_and_profile_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "profile_summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            profiles = FORMAL_PROFILES
            argv = [
                "run_profile_test.py",
                "--config",
                str(config_path),
                "--output",
                str(profile_path),
                "--summary-output",
                str(summary_path),
            ]

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in profiles])
                print(json.dumps({"output": args.output, "rows": len(profiles), "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            with (
                patch("run_util.profile_test.build_profile_table", side_effect=fake_build),
                patch("sys.argv", argv),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = run_profile_test_main()

            self.assertEqual(code, 0)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["section"], "experiment")
            self.assertEqual({row["section"] for row in summary_rows}, {"experiment", "profile"})
            self.assertNotIn("policy", {row["section"] for row in summary_rows})
            self.assertNotIn("policy_rows", summary_rows[0])
            full_gpu = next(row for row in summary_rows if row["section"] == "profile" and row["name"] == "full_gpu")
            self.assertEqual(full_gpu["count"], "1.0")
            self.assertNotEqual(full_gpu["count"], "1.0 from stdout")
            import run_util.profile_test as run_profile_test

            self.assertFalse(hasattr(run_profile_test, "run_policies"))

    def test_run_profile_test_surfaces_profile_validation_failure_without_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "profile_summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            argv = [
                "run_profile_test.py",
                "--config",
                str(config_path),
                "--output",
                str(profile_path),
                "--summary-output",
                str(summary_path),
            ]

            def fake_build(args: argparse.Namespace) -> int:
                rows = [_measurement("e1", profile, 0.0, measured=(profile != "kivi_4bit_residual32")).to_row() for profile in FORMAL_PROFILES]
                write_csv(Path(args.output), rows)
                print(json.dumps({"output": args.output, "rows": len(rows)}))
                return 0

            with (
                patch("run_util.profile_test.build_profile_table", side_effect=fake_build),
                patch("sys.argv", argv),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = run_profile_test_main()

            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertIn("profile 运行失败", summary_rows[0]["error"])
            self.assertIn("measured=False", summary_rows[0]["error"])
            self.assertEqual({row["section"] for row in summary_rows}, {"experiment", "profile"})
            self.assertTrue(any(row["section"] == "profile" and row["name"] == "kivi_4bit_residual32" for row in summary_rows))

    def test_run_profile_test_passes_require_ttft_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "profile_summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path, require_ttft=True)
            argv = [
                "run_profile_test.py",
                "--config",
                str(config_path),
                "--output",
                str(profile_path),
                "--summary-output",
                str(summary_path),
            ]

            def fake_build(args: argparse.Namespace) -> int:
                rows = [
                    ProfileMeasurement(
                        request_id="e1",
                        profile=profile,
                        adapter="full",
                        ok=True,
                        measured=True,
                        output_text="x",
                        latency_ms=2.0,
                        ttft_ms=None,
                        peak_memory_mib=1.0,
                        kv_cache_memory_mib=1.0,
                        resident_memory_mib=1.0,
                        quality_loss=0.0,
                        extra={"task": "qa", "length_bucket": "short", "split": "eval"},
                    ).to_row()
                    for profile in FORMAL_PROFILES
                ]
                write_csv(Path(args.output), rows)
                print(json.dumps({"output": args.output, "rows": len(rows)}))
                return 0

            with (
                patch("run_util.profile_test.build_profile_table", side_effect=fake_build),
                patch("sys.argv", argv),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = run_profile_test_main()

            self.assertEqual(code, 2)
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertIn("ttft_ms", summary_rows[0]["error"])
            self.assertEqual({row["section"] for row in summary_rows}, {"experiment", "profile"})

    def test_run_profile_test_accepts_h2o_only_subset_without_full_quality_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "profile_summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            argv = [
                "run_profile_test.py",
                "--config",
                str(config_path),
                "--adapters",
                "h2o",
                "--output",
                str(profile_path),
                "--summary-output",
                str(summary_path),
            ]

            def fake_build(args: argparse.Namespace) -> int:
                rows = [
                    ProfileMeasurement(
                        request_id="r1",
                        profile=profile,
                        adapter="h2o",
                        ok=True,
                        measured=True,
                        output_text="continued",
                        latency_ms=2.0,
                        ttft_ms=1.0,
                        peak_memory_mib=2.0,
                        kv_cache_memory_mib=2.0,
                        resident_memory_mib=2.0,
                        extra={"task": "summary", "length_bucket": "short", "split": "eval", "backend": "qwen2_h2o"},
                    ).to_row()
                    for profile in ("h2o_heavy10_recent10", "h2o_heavy15_recent15", "h2o_heavy20_recent20")
                ]
                write_csv(Path(args.output), rows)
                print(json.dumps({"output": args.output, "rows": len(rows)}))
                return 0

            with (
                patch("run_util.profile_test.build_profile_table", side_effect=fake_build),
                patch("sys.argv", argv),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = run_profile_test_main()

            self.assertEqual(code, 0)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["section"], "experiment")
            self.assertEqual(summary_rows[0]["ok"], "True")

    def test_run_policies_rejects_dry_run_replay_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dry.csv"
            write_csv(
                path,
                [
                    {
                        "request_id": "r1",
                        "profile": "engine_full_lru",
                        "adapter": "vllm_lru",
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

    def test_run_command_does_not_catch_keyerror_or_indexerror(self) -> None:
        with self.assertRaises(KeyError):
            run_command(lambda args: (_ for _ in ()).throw(KeyError("bug")), argparse.Namespace())
        with self.assertRaises(IndexError):
            run_command(lambda args: (_ for _ in ()).throw(IndexError("bug")), argparse.Namespace())

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
                    kv_cache_memory_mib=row.kv_cache_memory_mib,
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
            with patch("run_util.run_policies.build_policies", return_value=[BrokenPolicy()]):
                code = run_policies(args)
            self.assertEqual(code, 1)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["ok"] == "False" for record in records))
            self.assertIn("cannot decide e1", records[0]["error"])

    def test_policy_replay_missing_profile_records_failure_and_continues_other_policy(self) -> None:
        class MissingProfilePolicy(Policy):
            name = "missing_profile_policy"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                return Action(profile="kivi_4bit", reason="missing profile test")

        class ExactPolicy(Policy):
            name = "exact_policy"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                return Action(profile="full_gpu", reason="exact test")

        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            rows = [
                ProfileMeasurement(
                    request_id="e1",
                    profile="full_gpu",
                    adapter="full",
                    ok=True,
                    measured=True,
                    output_text="x",
                    latency_ms=1.0,
                    ttft_ms=1.0,
                    peak_memory_mib=1.0,
                    kv_cache_memory_mib=1.0,
                    quality_loss=0.0,
                    extra={"task": "qa", "length_bucket": "short", "split": "eval"},
                )
            ]
            write_csv(measurements_path, [row.to_row() for row in rows])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["missing_profile_policy", "exact_policy"],
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=None,
                use_pandas_replay=False,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with patch("run_util.run_policies.build_policies", return_value=[MissingProfilePolicy(), ExactPolicy()]), redirect_stdout(stream):
                code = run_policies(args)
            self.assertEqual(code, 1)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([record["policy"] for record in records], ["missing_profile_policy", "exact_policy"])
            self.assertEqual(records[0]["ok"], "False")
            self.assertIn("缺少回放数据", records[0]["error"])
            self.assertEqual(records[1]["ok"], "True")

    def test_with_quality_uses_full_gpu_baseline_only(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta", extra={"task": "qa", "reference": "alpha beta"}),
            ProfileMeasurement("r1", "kivi_4bit_residual32", "kivi", True, True, output_text="alpha", extra={"task": "qa", "reference": "alpha beta"}),
        ]
        updated = with_quality(rows, {"full_gpu"})
        by_profile = {row.profile: row for row in updated}
        self.assertEqual(by_profile["full_gpu"].quality_loss, 0.0)
        self.assertLess(by_profile["kivi_4bit_residual32"].quality_loss, 1.0)

    def test_with_quality_requires_full_gpu_for_lossy_quality(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta", extra={"task": "qa"}),
            ProfileMeasurement("r1", "kivi_4bit_residual32", "kivi", True, True, output_text="alpha", extra={"task": "qa"}),
        ]
        updated = with_quality(rows, {"full_gpu"})
        by_profile = {row.profile: row for row in updated}
        self.assertEqual(by_profile["full_gpu"].quality_loss, 0.0)
        self.assertIsNotNone(by_profile["kivi_4bit_residual32"].quality_loss)

    def test_quality_metrics_tolerate_none_but_compute_missing_as_full_loss(self) -> None:
        self.assertEqual(normalized_exact_match_loss(None, ""), 0.0)
        self.assertEqual(token_f1_loss(None, ""), 0.0)
        self.assertEqual(rouge_l_loss(None, ""), 0.0)
        loss, metrics = compute_quality_loss("qa", None, "answer")
        self.assertEqual(loss, 1.0)
        self.assertEqual(metrics, {"em": 1.0, "f1": 1.0, "rouge_l": 1.0})
        loss, metrics = compute_quality_loss("summary", "candidate", None)
        self.assertEqual(loss, 1.0)
        self.assertEqual(metrics["rouge_l"], 1.0)
        loss, metrics = compute_quality_loss("unknown", "same", "same")
        self.assertEqual(loss, 0.0)
        self.assertEqual(metrics["em"], 0.0)

    def test_select_primary_loss_uses_f1_for_qa_long_context(self) -> None:
        self.assertEqual(select_primary_loss("qa_long_context"), "f1")

    def test_select_primary_loss_rejects_unknown_task_in_strict_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_primary_loss("unknown", strict=True)

    def test_quality_metrics_normalize_case_whitespace_and_punctuation_noise(self) -> None:
        loss, metrics = compute_quality_loss("summary", " The baselines!!!! ", "the baselines")
        self.assertEqual(loss, 0.0)
        self.assertEqual(metrics["em"], 0.0)
        self.assertEqual(metrics["rouge_l"], 0.0)

    def test_annotate_measurement_copies_reference_into_extra_reference(self) -> None:
        measurement = ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="answer")
        request = Request("r1", "qa", "prompt", reference="gold answer", metadata={"split": "eval"})

        annotated = annotate_measurement(measurement, request, False)

        self.assertEqual(annotated.extra["reference"], "gold answer")

    def test_with_quality_writes_primary_metric_for_exact_and_lossy_rows(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta", extra={"task": "qa_long_context", "reference": "alpha beta"}),
            ProfileMeasurement("r1", "kivi_4bit", "kivi", True, True, output_text="alpha", extra={"task": "qa_long_context", "reference": "alpha beta"}),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"})}

        self.assertEqual(updated["full_gpu"].extra["primary_metric"], "f1")
        self.assertEqual(updated["kivi_4bit"].extra["primary_metric"], "f1")
        self.assertAlmostEqual(updated["kivi_4bit"].extra["metric_loss_f1"], 1.0 / 3.0)

    def test_dry_profile_measurement_marks_synthetic_source(self) -> None:
        row = profiles_base.dry_profile_measurement(
            "full",
            Request("r1", "qa", "prompt"),
            ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True),
            latency_ms=1.0,
            peak_memory_mib=2.0,
        )

        self.assertEqual(row.extra["dry_run"], "true")
        self.assertEqual(row.extra["source"], "synthetic_schema_check")
        self.assertEqual(row.extra["backend"], "synthetic")
        self.assertEqual(row.extra["ttft_semantics"], "unavailable")

    def test_with_quality_uses_full_gpu_and_candidate_scores_against_same_reference(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta gamma", extra={"task": "qa", "reference": "alpha beta gamma"}),
            ProfileMeasurement("r1", "kivi_4bit", "kivi", True, True, output_text="alpha", extra={"task": "qa", "reference": "alpha beta gamma"}),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"})}

        self.assertAlmostEqual(updated["kivi_4bit"].quality_loss, 0.5)
        self.assertAlmostEqual(updated["kivi_4bit"].quality_score, 0.5)

    def test_with_quality_scores_short_unquantized_kivi_rows_when_measured(self) -> None:
        rows = [
            ProfileMeasurement(
                request_id="r1",
                profile="full_gpu",
                adapter="full",
                ok=True,
                measured=True,
                output_text="reference answer",
                latency_ms=1.0,
                ttft_ms=1.0,
                peak_memory_mib=1.0,
                resident_memory_mib=1.0,
                extra={"task": "summary", "length_bucket": "short", "split": "calibration", "reference": "reference answer"},
            ),
            ProfileMeasurement(
                request_id="r1",
                profile="kivi_4bit_residual64",
                adapter="kivi",
                ok=True,
                measured=True,
                output_text="reference answer",
                latency_ms=2.0,
                ttft_ms=2.0,
                peak_memory_mib=2.0,
                resident_memory_mib=2.0,
                extra={
                    "task": "summary",
                    "length_bucket": "short",
                    "split": "calibration",
                    "reference": "reference answer",
                    "kivi_quantization_triggered": False,
                    "kivi_effective_mode": "unquantized_short_request",
                },
            ),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"})}

        self.assertEqual(updated["kivi_4bit_residual64"].quality_loss, 0.0)
        self.assertEqual(updated["kivi_4bit_residual64"].quality_score, 1.0)

    def test_build_profile_table_accepts_measured_unquantized_kivi_chunk(self) -> None:
        requests = [Request("r1", "summary", "prompt", reference="answer", metadata={"split": "calibration"})]

        class StubAdapter:
            def __init__(self, name):
                self.name = name

            def profiles(self):
                if self.name == "full":
                    return (ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True),)
                return (ProfileSpec("kivi_4bit_residual64", "kivi", "edgekv-kivi", lossy=True),)

            def profile_many(self, request_chunk, profile_name, dry_run=True):
                if profile_name == "full_gpu":
                    return [
                        ProfileMeasurement(
                            request_id="r1",
                            profile="full_gpu",
                            adapter="full",
                            ok=True,
                            measured=True,
                            output_text="answer",
                            latency_ms=1.0,
                            ttft_ms=1.0,
                            peak_memory_mib=1.0,
                            kv_cache_memory_mib=1.0,
                            resident_memory_mib=1.0,
                            extra={
                                "task": "summary",
                                "length_bucket": "short",
                                "split": "calibration",
                                "reference": "answer",
                            },
                        )
                    ]
                return [
                    ProfileMeasurement(
                        request_id="r1",
                        profile="kivi_4bit_residual64",
                        adapter="kivi",
                        ok=True,
                        measured=True,
                        output_text="answer",
                        latency_ms=2.0,
                        ttft_ms=2.0,
                        peak_memory_mib=2.0,
                        kv_cache_memory_mib=2.0,
                        resident_memory_mib=2.0,
                        extra={
                            "task": "summary",
                            "length_bucket": "short",
                            "split": "calibration",
                            "reference": "answer",
                            "kivi_quantization_triggered": False,
                            "kivi_effective_mode": "unquantized_short_request",
                        },
                    )
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full", "kivi"], "names": ["full_gpu", "kivi_4bit_residual64"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["full", "kivi"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu", "kivi_4bit_residual64"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[StubAdapter("full"), StubAdapter("kivi")]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_util.build_profile_table.exact_profiles", return_value={"full_gpu"}),
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

            self.assertEqual(code, 0)
            rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))

        by_profile = {row["profile"]: row for row in rows}
        self.assertEqual(by_profile["kivi_4bit_residual64"]["measured"], "True")
        self.assertEqual(by_profile["kivi_4bit_residual64"]["extra_kivi_effective_mode"], "unquantized_short_request")

    def test_build_profile_table_accepts_h2o_only_subset_without_full_quality_baseline(self) -> None:
        requests = [Request("r1", "summary", "continue", reference="answer", metadata={"split": "calibration"})]

        class StubH2OAdapter:
            name = "h2o"

            def profiles(self):
                return (ProfileSpec("h2o_heavy10_recent10", "h2o", "edgekv-h2o", lossy=True),)

            def profile_many(self, request_chunk, profile_name, dry_run=True):
                return [
                    ProfileMeasurement(
                        request_id="r1",
                        profile="h2o_heavy10_recent10",
                        adapter="h2o",
                        ok=True,
                        measured=True,
                        output_text="continued",
                        latency_ms=2.0,
                        ttft_ms=1.0,
                        peak_memory_mib=2.0,
                        kv_cache_memory_mib=2.0,
                        resident_memory_mib=2.0,
                        extra={
                            "task": "summary",
                            "length_bucket": "short",
                            "split": "calibration",
                            "reference": "answer",
                            "backend": "qwen2_h2o",
                            "h2o_prune_events": 3,
                        },
                    )
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            with (
                patch("run_util.build_profile_table.load_config", return_value={"profiles": {"adapters": ["full", "h2o"], "names": ["full_gpu", "h2o_heavy10_recent10"]}}),
                patch("run_util.build_profile_table.config_adapters", return_value=["full", "h2o"]),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu", "h2o_heavy10_recent10"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[StubH2OAdapter()]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_util.build_profile_table.exact_profiles", return_value={"full_gpu"}),
            ):
                code = build_profile_table(
                    argparse.Namespace(
                        config="config.yaml",
                        adapters=["h2o"],
                        output=str(output_path),
                        import_measurements="",
                        dry_run=False,
                    )
                )

            self.assertEqual(code, 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["profile"], "h2o_heavy10_recent10")
            self.assertEqual(rows[0]["quality_loss"], "")

    def test_with_quality_skips_lossy_quality_when_reference_missing_in_session_mode(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="The baselines", extra={"task": "summary"}),
            ProfileMeasurement("r1", "h2o_heavy10_recent10", "h2o", True, True, output_text="the baselines!!!!", extra={"task": "summary"}),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"}, quality_mode="session_diagnostic")}

        self.assertIsNone(updated["h2o_heavy10_recent10"].quality_loss)
        self.assertIsNone(updated["h2o_heavy10_recent10"].quality_score)

    def test_with_quality_rejects_missing_reference_in_baseline_mode(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="The baselines", extra={"task": "summary"}),
            ProfileMeasurement("r1", "h2o_heavy10_recent10", "h2o", True, True, output_text="the baselines!!!!", extra={"task": "summary"}),
        ]

        with self.assertRaisesRegex(ValueError, "baseline quality smoke"):
            with_quality(rows, {"full_gpu"}, quality_mode="baseline")

    def test_build_profile_table_rejects_chat_without_reference_for_baseline_quality(self) -> None:
        requests = [Request("s1_t0", "chat", "hello", metadata={"split": "eval"})]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            with (
                patch("run_util.build_profile_table.load_config", return_value={
                    "profiles": {"adapters": ["full"], "names": ["full_gpu"]},
                    "data": {"quality_mode": "baseline"},
                }),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1, "max_requests": 0}),
                patch("run_util.build_profile_table.config_adapters", return_value=["full"]),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
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

        self.assertEqual(code, 2)

    def test_build_profile_table_rejects_unknown_task_for_baseline_quality(self) -> None:
        requests = [Request("r1", "classification", "label this", reference="positive", metadata={"split": "eval"})]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            with (
                patch("run_util.build_profile_table.load_config", return_value={
                    "profiles": {"adapters": ["full"], "names": ["full_gpu"]},
                    "data": {"quality_mode": "baseline"},
                }),
                patch("run_util.build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_util.build_profile_table.config_runtime", return_value={"repeat": 1, "max_requests": 0}),
                patch("run_util.build_profile_table.config_adapters", return_value=["full"]),
                patch("run_util.build_profile_table.build_profile_adapters", return_value=[]),
                patch("run_util.build_profile_table.load_requests", return_value=(requests, False)),
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

        self.assertEqual(code, 2)

    def test_with_quality_marks_missing_candidate_as_full_loss_and_requires_full_gpu_baseline(self) -> None:
        rows = [
            _measurement("r1", "full_gpu", 0.0),
            ProfileMeasurement(
                request_id="r1",
                profile="kivi_4bit",
                adapter="kivi",
                ok=True,
                measured=True,
                output_text="",
                ttft_ms=1.0,
                peak_memory_mib=1.0,
                quality_loss=None,
                extra={"task": "qa", "length_bucket": "short", "split": "eval"},
            ),
            ProfileMeasurement(
                request_id="r2",
                profile="kivi_4bit",
                adapter="kivi",
                ok=True,
                measured=True,
                output_text="",
                ttft_ms=1.0,
                peak_memory_mib=1.0,
                quality_loss=None,
                extra={"task": "qa", "length_bucket": "short", "split": "eval", "reference": "answer"},
            ),
        ]
        updated = {row.request_id: row for row in with_quality(rows, {"full_gpu"})}
        self.assertEqual(updated["r1"].quality_loss, 1.0)
        self.assertEqual(updated["r1"].quality_score, 0.0)
        self.assertIsNone(updated["r2"].quality_loss)
        self.assertIsNone(updated["r2"].quality_score)

    def test_aal_wilson_marks_and_releases_unsafe_profile(self) -> None:
        state = AALState(epsilon=0.2, delta=0.05, window_size=5)
        for index in range(5):
            state.record(AuditSample(f"bad{index}", "kivi_4bit", 0.1, 0.9, 0.2))
        self.assertIn("kivi_4bit", state.unsafe_profiles)
        state.calibration_update("kivi_4bit")
        self.assertNotIn("kivi_4bit", state.unsafe_profiles)

    def test_wilson_empty_window_is_stable_zero_upper_bound(self) -> None:
        detector = WilsonDriftDetector(epsilon=0.2, delta=0.05)
        self.assertEqual(detector.wilson_upper(), 0.0)


if __name__ == "__main__":
    unittest.main()
