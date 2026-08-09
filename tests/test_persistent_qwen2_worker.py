from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from profiles.base import PersistentProfileWorker
from profiles import qwen2_kv_runtime


def _payload(prompt: str, turn_index: int) -> dict[str, object]:
    return {
        "adapter": "full",
        "profile": "full_gpu",
        "requests": [
            {
                "request_id": f"r{turn_index}",
                "session_id": "s1",
                "turn_index": turn_index,
                "task": "qa",
                "prompt": prompt,
                "history_turns": [],
                "profile": "full_gpu",
                "model_name": "/tmp/model",
                "max_new_tokens": 4,
            }
        ],
        "session_runtime_state": {"sessions": {}},
    }


def test_worker_run_batch_reuses_runtime_between_batches() -> None:
    worker_state: dict[str, object] = {}
    runtime = {"runtime_id": "sentinel"}

    with (
        patch("profiles.qwen2_kv_runtime._prepare_full_runtime", return_value=runtime) as prepare_runtime,
        patch(
            "profiles.qwen2_kv_runtime._run_full_request",
            side_effect=[
                {
                    "ok": True,
                    "measured": True,
                    "output_text": "a",
                    "latency_ms": 1.0,
                    "ttft_ms": 1.0,
                    "peak_memory_mib": 2.0,
                    "kv_cache_memory_mib": 2.0,
                    "resident_memory_mib": 2.0,
                },
                {
                    "ok": True,
                    "measured": True,
                    "output_text": "b",
                    "latency_ms": 1.0,
                    "ttft_ms": 1.0,
                    "peak_memory_mib": 3.0,
                    "kv_cache_memory_mib": 3.0,
                    "resident_memory_mib": 3.0,
                },
            ],
        ),
    ):
        init_result = qwen2_kv_runtime.worker_init({"adapter": "full", "runtime_config": {}}, worker_state)
        first = qwen2_kv_runtime.worker_run_batch(_payload("hello", 0), worker_state)
        second = qwen2_kv_runtime.worker_run_batch(_payload("world", 1), worker_state)

    assert init_result["ok"] is True
    assert prepare_runtime.call_count == 1
    assert first["worker"]["mode"] == "persistent"
    assert second["worker"]["mode"] == "persistent"
    assert second["session_runtime_state"]["sessions"]["s1"]["resident_gpu_mib"] == 3.0


def test_worker_run_batch_reports_fatal_error_and_releases_runtime() -> None:
    worker_state: dict[str, object] = {}
    runtime = {"runtime_id": "sentinel"}

    with (
        patch("profiles.qwen2_kv_runtime._prepare_full_runtime", return_value=runtime),
        patch(
            "profiles.qwen2_kv_runtime._run_full_request",
            return_value={
                "ok": False,
                "measured": False,
                "error": "CUDA out of memory",
                "failure_stage": "generate",
            },
        ),
        patch("profiles.qwen2_kv_runtime._release_runtime_resources") as release_runtime,
    ):
        qwen2_kv_runtime.worker_init({"adapter": "full", "runtime_config": {}}, worker_state)
        result = qwen2_kv_runtime.worker_run_batch(_payload("hello", 0), worker_state)

    assert result["ok"] is False
    assert "fatal_error" in result
    release_runtime.assert_called_once()


def test_persistent_worker_starts_with_env_python_instead_of_conda_run() -> None:
    worker = PersistentProfileWorker(
        adapter="full",
        env_name="tailguardkv-base",
        runtime_module="profiles.qwen2_kv_runtime",
        runtime_config={"timeout_s": 30},
    )
    proc = SimpleNamespace(stdin=None, stdout=None, stderr=None, poll=lambda: None)

    with (
        patch.dict("os.environ", {"CONDA_EXE": "/opt/miniforge3/bin/conda"}, clear=False),
        patch("profiles.base.Path.exists", return_value=True),
        patch("profiles.base.subprocess.Popen", return_value=proc) as popen,
        patch.object(PersistentProfileWorker, "request", return_value={"ok": True}),
    ):
        worker.start()

    command = popen.call_args.args[0]
    assert command[:3] == ["/opt/miniforge3/envs/tailguardkv-base/bin/python", "-m", "profiles.persistent_worker"]
    assert "conda" not in command[0]


def test_release_runtime_resources_drops_model_refs_before_empty_cache() -> None:
    class FakeCuda:
        def __init__(self, runtime: dict[str, object]) -> None:
            self.runtime = runtime

        def empty_cache(self) -> None:
            assert "model" not in self.runtime
            assert "tokenizer" not in self.runtime

        def synchronize(self, gpu_index: int) -> None:
            del gpu_index
            return None

        def device_count(self) -> int:
            return 1

        def is_available(self) -> bool:
            return True

    runtime: dict[str, object] = {
        "model": object(),
        "tokenizer": object(),
        "device": object(),
        "session_reuse": {},
    }
    fake_torch = SimpleNamespace(cuda=FakeCuda(runtime))
    runtime["torch"] = fake_torch

    qwen2_kv_runtime._release_runtime_resources(runtime)

    assert runtime == {}
