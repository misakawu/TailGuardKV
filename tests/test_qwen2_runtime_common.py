from __future__ import annotations

from types import SimpleNamespace

import pytest

from profiles import qwen2_runtime_common
from profiles.qwen2_runtime_common import load_qwen2_model


def test_load_qwen2_model_builds_explicit_two_gpu_device_map_for_odd_layers() -> None:
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

    class FakeTorch:
        float16 = "float16"
        cuda = FakeCuda()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured.update(kwargs)

            class _Model:
                config = SimpleNamespace(num_hidden_layers=5)
                model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(device="cuda:0")))

                @staticmethod
                def eval() -> None:
                    return None

            return _Model()

    load_qwen2_model(
        {"model_name": "/tmp/model", "local_files_only": True, "num_hidden_layers": 5},
        FakeTorch(),
        FakeModel,
        FakeTokenizer,
    )

    assert captured["max_memory"] == {0: "8704MiB", 1: "9216MiB"}
    assert captured["device_map"] == {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 1,
        "model.layers.4": 1,
        "model.norm": 1,
        "lm_head": 1,
    }


def test_load_qwen2_model_builds_explicit_two_gpu_device_map_for_even_layers() -> None:
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

    class FakeTorch:
        float16 = "float16"
        cuda = FakeCuda()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured.update(kwargs)

            class _Model:
                config = SimpleNamespace(num_hidden_layers=4)
                model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(device="cuda:0")))

                @staticmethod
                def eval() -> None:
                    return None

            return _Model()

    load_qwen2_model(
        {"model_name": "/tmp/model", "local_files_only": True, "num_hidden_layers": 4},
        FakeTorch(),
        FakeModel,
        FakeTokenizer,
    )

    assert captured["device_map"]["model.layers.0"] == 0
    assert captured["device_map"]["model.layers.1"] == 0
    assert captured["device_map"]["model.layers.2"] == 1
    assert captured["device_map"]["model.layers.3"] == 1
    assert set(captured["device_map"].values()) == {0, 1}


def test_load_qwen2_model_supports_single_gpu_device_strategy() -> None:
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

    class FakeTorch:
        float16 = "float16"
        cuda = FakeCuda()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured.update(kwargs)

            class _Model:
                config = SimpleNamespace(num_hidden_layers=4)
                model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(device="cuda:0")))

                @staticmethod
                def eval() -> None:
                    return None

            return _Model()

    load_qwen2_model(
        {
            "model_name": "/tmp/model",
            "local_files_only": True,
            "num_hidden_layers": 4,
            "device_strategy": "single_gpu",
        },
        FakeTorch(),
        FakeModel,
        FakeTokenizer,
    )

    assert captured["max_memory"] == {0: "8704MiB"}
    assert set(captured["device_map"].values()) == {0}


def test_safe_max_memory_mib_uses_asymmetric_dual_gpu_reserves() -> None:
    class FakeCuda:
        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

    class FakeTorch:
        cuda = FakeCuda()

    assert qwen2_runtime_common._safe_max_memory_mib(FakeTorch(), 0) == 8704
    assert qwen2_runtime_common._safe_max_memory_mib(FakeTorch(), 1) == 9216


def test_generate_decode_reports_combined_peak_memory_across_gpu_0_and_1(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str) -> "FakeTensor":
            return self

    class FakeTokenizer:
        eos_token_id = 7

        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, FakeTensor]:
            return {"input_ids": FakeTensor("ids"), "attention_mask": FakeTensor("mask")}

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            return None

        @staticmethod
        def synchronize(device: object) -> None:
            return None

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            return {0: 3 * 1024**2, 1: 5 * 1024**2}[int(device)]

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            return (8 * 1024**2, 11_264 * 1024**2)

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode() -> FakeInferenceMode:
            return FakeInferenceMode()

    monkeypatch.setattr(
        qwen2_runtime_common,
        "generate_with_first_token_timing",
        lambda *args, **kwargs: {
            "output_text": "ok",
            "ttft_ms": 4.0,
            "stage_generate_ms": 5.0,
            "stage_prefill_ms": 2.0,
            "stage_first_token_ms": 3.0,
            "past_key_values": None,
        },
    )

    result = qwen2_runtime_common.generate_decode(
        object(),
        FakeTokenizer(),
        "cuda:0",
        {"prompt": "hello", "max_new_tokens": 2},
        FakeTorch(),
        stage_startup_ms=1.0,
        stage_model_load_ms=2.0,
        worker_mode="batch",
    )

    assert result["ok"] is True
    assert result["peak_memory_mib"] == pytest.approx(8.0)
    assert result["gpu0_peak_memory_mib"] == pytest.approx(3.0)
    assert result["gpu1_peak_memory_mib"] == pytest.approx(5.0)


def test_generate_decode_includes_dual_gpu_snapshot_when_cuda_oom_occurs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str) -> "FakeTensor":
            return self

    class FakeTokenizer:
        eos_token_id = 7

        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, FakeTensor]:
            return {"input_ids": FakeTensor("ids"), "attention_mask": FakeTensor("mask")}

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            return None

        @staticmethod
        def synchronize(device: object) -> None:
            return None

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            return {0: 9 * 1024**2, 1: 10 * 1024**2}[int(device)]

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            return {
                0: (900 * 1024**2, 11_264 * 1024**2),
                1: (700 * 1024**2, 11_264 * 1024**2),
            }[int(index)]

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode() -> FakeInferenceMode:
            return FakeInferenceMode()

    def fake_generate(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.21 GiB")

    monkeypatch.setattr(qwen2_runtime_common, "generate_with_first_token_timing", fake_generate)

    result = qwen2_runtime_common.generate_decode(
        object(),
        FakeTokenizer(),
        "cuda:0",
        {"prompt": "hello", "max_new_tokens": 2},
        FakeTorch(),
        worker_mode="batch",
    )

    assert result["ok"] is False
    assert result["failure_stage"] == "generate"
    assert result["gpu0_free_mib"] == pytest.approx(900.0)
    assert result["gpu1_free_mib"] == pytest.approx(700.0)
    assert result["gpu0_used_mib"] == pytest.approx(10_364.0)
    assert result["gpu1_used_mib"] == pytest.approx(10_564.0)


def test_release_runtime_cuda_resources_clears_objects_runs_gc_and_empties_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []

    class Clearable:
        def clear(self) -> None:
            events.append("clear")

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

        @staticmethod
        def synchronize(device: object) -> None:
            events.append(("sync", int(device)))

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=11_264 * 1024**2)

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(qwen2_runtime_common.gc, "collect", lambda: events.append("gc"))

    qwen2_runtime_common.release_runtime_cuda_resources(FakeTorch(), Clearable(), object())

    assert events[0] == "clear"
    assert "gc" in events
    assert "empty_cache" in events
    assert ("sync", 0) in events
    assert ("sync", 1) in events
