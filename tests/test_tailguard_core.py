from __future__ import annotations

import argparse
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

from calibration.conformal import ConformalGuard
from aal import AALState, AuditSample, WilsonDriftDetector
from backends.measured_replay import MeasuredReplayBackend
from core_types import Action
from core_types import PolicyRunRecord, ProfileMeasurement, ProfileSpec, Request
from experiment_common import annotate_measurement, config_adapters, config_policies, config_profiles, config_runtime, exact_profiles, failed_measurement_summary, limit_requests_by_split, load_config, validate_profile_measurements, with_quality, write_csv
from metrics.quality import compute_quality_loss, normalized_exact_match_loss, rouge_l_loss, token_f1_loss
from metrics import MetricCollector
from policies.base import Policy
from policies.registry import build_policies
from profiles import base as profiles_base
from profiles.base import qwen2_kv_profile_many_measurements, qwen2_kv_profile_measurement, transformers_profile_many_measurements, transformers_profile_measurement
from profiles.full import FullKVAdapter
from profiles.h2o import H2OAdapter
from profiles.kivi import KIVIAdapter
from profiles.kivi_cache import KIVICache
from profiles import qwen2_kv_runtime
from profiles.registry import build_profile_adapters
from vllm_lru_policy import create_vllm_policy
from env_asset_prepare.prepare_pilot_assets import format_longbench_prompt
import run_build_profile_table as profile_table_module
from run_build_profile_table import build_profile_table
from run_cli_common import run_command
from run_experiment import _policy_output_for_sweep, _policy_sweep_points, _summary_rows, build_parser as build_experiment_parser, pilot_smoke_measured
from run_profile_test import build_parser as build_profile_test_parser, main as run_profile_test_main
from run_run_policies import run_policies


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


def _write_pilot_test_config(path: Path, profile_output: Path, policy_output: Path, summary_output: Path) -> None:
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
                "  names: [full_lru, static_best, static_safe, utility_dynamic, uncalibrated_dynamic]",
                "pilot:",
                "  epsilons: [0.05]",
                "  deltas: [0.05]",
                "  memory_budgets_mib: [4900]",
                "outputs:",
                f"  smoke_profiles: {profile_output}",
                f"  smoke_policy: {policy_output}",
                f"  smoke_summary: {summary_output}",
            ]
        ),
        encoding="utf-8",
    )


class TailGuardCoreTest(unittest.TestCase):
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

        with patch("profiles.full.transformers_profile_many_measurements") as many:
            many.return_value = [_measurement("r1", "full_gpu", 0.0)]
            rows = adapter.profile_many(requests, "full_gpu", dry_run=False)

        many.assert_called_once()
        self.assertEqual(rows[0].request_id, "r1")

    def test_kivi_adapter_measured_profile_many_uses_batch_runtime(self) -> None:
        adapter = KIVIAdapter({"max_new_tokens": 4})
        requests = [Request(request_id="r1", task="summary", prompt="a")]

        with patch("profiles.kivi.qwen2_kv_profile_many_measurements") as many:
            many.return_value = [_measurement("r1", "kivi_4bit_residual32", None)]
            rows = adapter.profile_many(requests, "kivi_4bit_residual32", dry_run=False)

        many.assert_called_once()
        self.assertEqual(rows[0].profile, "kivi_4bit_residual32")

    def test_h2o_adapter_measured_profile_many_uses_batch_runtime(self) -> None:
        adapter = H2OAdapter({"max_new_tokens": 4})
        requests = [Request(request_id="r1", task="summary", prompt="a")]

        with patch("profiles.h2o.qwen2_kv_profile_many_measurements") as many:
            many.return_value = [_measurement("r1", "h2o_heavy10_recent10", None)]
            rows = adapter.profile_many(requests, "h2o_heavy10_recent10", dry_run=False)

        many.assert_called_once()
        self.assertEqual(rows[0].profile, "h2o_heavy10_recent10")

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
                extra={"task": "summary", "length_bucket": "short", "split": "calibration", "stage_total_ms": 9.0},
            )

            profile_table_module._append_profile_rows(output, [row.to_row()])
            text = output.read_text(encoding="utf-8")

        self.assertIn("extra_stage_total_ms", text)

    def test_qwen2_runtime_profiles_use_generate_instead_of_manual_decode(self) -> None:
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
                return self.values[index]

        class FakeGenerateOutput:
            def __init__(self, sequences):
                self.sequences = sequences

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                return {
                    "input_ids": FakeTensor([11, 12, 13, 14, 15, 16]),
                    "attention_mask": FakeTensor([1, 1, 1, 1, 1, 1]),
                }

            def decode(self, tokens, skip_special_tokens=True):
                return "generated output"

        class FakeModel:
            def __init__(self):
                self.generate_calls = []
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

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return FakeGenerateOutput(FakeTensor([[11, 12, 13, 14, 15, 16, 99, 100]]))

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
                tracker["h2o_prune_events"] = 1

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
        self.assertEqual(len(kivi_model.generate_calls), 1)
        self.assertEqual(len(h2o_model.generate_calls), 1)
        self.assertEqual(kivi_model.generate_calls[0]["max_new_tokens"], 2)
        self.assertEqual(h2o_model.generate_calls[0]["max_new_tokens"], 2)
        self.assertIn("past_key_values", kivi_model.generate_calls[0])
        self.assertNotIsInstance(kivi_model.generate_calls[0]["past_key_values"], tuple)
        self.assertNotIn("past_key_values", h2o_model.generate_calls[0])

    def test_generate_decode_accepts_explicit_past_key_values(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def to(self, _device):
                return self

            def __getitem__(self, index):
                return self.values[index]

        class FakeGenerateOutput:
            def __init__(self, sequences):
                self.sequences = sequences

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors="pt"):
                return {
                    "input_ids": FakeTensor([11, 12, 13]),
                    "attention_mask": FakeTensor([1, 1, 1]),
                }

            def decode(self, tokens, skip_special_tokens=True):
                return "generated output"

        class FakeModel:
            def __init__(self):
                self.generate_kwargs = None

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return FakeGenerateOutput(FakeTensor([[11, 12, 13, 99]]))

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
        self.assertIs(model.generate_kwargs["past_key_values"], cache)

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

    def test_registry_rejects_quality_oracle_after_policy_cleanup(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "未知 policy: quality_oracle"):
            build_policies(
                ["quality_oracle"],
                rows,
                rows,
                ["full_gpu"],
                0.2,
                0.05,
                {"full_gpu"},
            )

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
        self.assertEqual(action.fallback_reason, "点预测阈值通过")

    def test_utility_dynamic_falls_back_to_full_gpu_when_lossy_is_unsafe(self) -> None:
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
        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.rejected_profile, "kivi_4bit_residual32")
        self.assertIsNotNone(action.rejected_pred_loss)
        self.assertIsNotNone(action.rejected_risk_upper)

    def test_uncalibrated_dynamic_falls_back_to_full_gpu_when_lossy_is_unsafe(self) -> None:
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
        self.assertEqual(action.rejected_profile, "kivi_4bit_residual32")

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
                "kivi_4bit_residual32",
                "kivi_4bit_residual64",
                "kivi_2bit_residual32",
                "kivi_2bit_residual64",
                "h2o_heavy10_recent10",
                "h2o_heavy15_recent15",
                "h2o_heavy20_recent20",
            ],
        )
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["bits"], 4)
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["kivi_residual_length"], 32)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["bits"], 2)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["kivi_residual_length"], 64)
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
        expected_profiles = [
            "full_gpu",
            "kivi_4bit_residual32",
            "kivi_4bit_residual64",
            "kivi_2bit_residual32",
            "kivi_2bit_residual64",
            "h2o_heavy10_recent10",
            "h2o_heavy15_recent15",
            "h2o_heavy20_recent20",
        ]
        self.assertEqual(config_adapters(pilot), ["full", "kivi", "h2o"])
        self.assertEqual(config_profiles(pilot), expected_profiles)
        self.assertEqual(config_profiles(pilot_50), expected_profiles)
        self.assertEqual(pilot["data"]["max_requests"], 200)
        self.assertEqual(pilot_50["data"]["max_requests"], 50)
        self.assertEqual(pilot["pilot"]["memory_budgets_mib"], [4900, 5000])
        self.assertEqual(pilot_50["pilot"]["memory_budgets_mib"], [4900, 5000])
        self.assertNotIn("profile_smoke_model", pilot["model"])
        self.assertNotIn("profile_smoke_model", pilot_50["model"])
        self.assertTrue(pilot["policies"]["record_rejected_unsafe"])
        self.assertTrue(pilot_50["policies"]["record_rejected_unsafe"])
        self.assertEqual(config_policies(pilot), ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"])
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
        self.assertEqual(backend.run([Request("r1", "qa", "p")], ["full_gpu"])[0], row)

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
            ttft_ms=1.0,
            peak_memory_mib=1.0,
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
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}, "data": {"requests": "data/fixtures/e0_reproduce_requests.jsonl"}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch(
                    "run_build_profile_table.load_requests",
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
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1, "max_requests": 20}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
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
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_build_profile_table._append_profile_rows", side_effect=record_append),
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
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
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
                            "rows": 5,
                            "epsilon": 0.05,
                            "delta": 0.05,
                            "memory_budget_mib": 6144.0,
                            "summary": {"full_lru": {"p95_ttft_ms": 1.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
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
            self.assertEqual(summary_rows[0]["policy_rows"], "5")
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
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
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
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertTrue(summary_path.exists())

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
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies") as policy_mock,
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
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies") as policy_mock,
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
                patch("run_profile_test.build_profile_table", side_effect=fake_build),
                patch("run_profile_test.run_policies") as policies,
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
            policies.assert_not_called()

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
                patch("run_profile_test.build_profile_table", side_effect=fake_build),
                patch("run_profile_test.run_policies") as policies,
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
            policies.assert_not_called()

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
                    ttft_ms=1.0,
                    peak_memory_mib=1.0,
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
            with patch("run_run_policies.build_policies", return_value=[MissingProfilePolicy(), ExactPolicy()]), redirect_stdout(stream):
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

    def test_with_quality_uses_full_gpu_and_candidate_scores_against_same_reference(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta gamma", extra={"task": "qa", "reference": "alpha beta gamma"}),
            ProfileMeasurement("r1", "kivi_4bit", "kivi", True, True, output_text="alpha", extra={"task": "qa", "reference": "alpha beta gamma"}),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"})}

        self.assertAlmostEqual(updated["kivi_4bit"].quality_loss, 0.5)
        self.assertAlmostEqual(updated["kivi_4bit"].quality_score, 0.5)

    def test_with_quality_falls_back_to_text_similarity_when_reference_missing(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="The baselines", extra={"task": "summary"}),
            ProfileMeasurement("r1", "h2o_heavy10_recent10", "h2o", True, True, output_text="the baselines!!!!", extra={"task": "summary"}),
        ]

        updated = {row.profile: row for row in with_quality(rows, {"full_gpu"})}

        self.assertEqual(updated["h2o_heavy10_recent10"].quality_loss, 0.0)
        self.assertEqual(updated["h2o_heavy10_recent10"].quality_score, 1.0)

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
