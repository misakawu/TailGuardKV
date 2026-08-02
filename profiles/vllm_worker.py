from __future__ import annotations

import json
import os
import resource
import time


def main() -> int:
    from vllm import LLM, SamplingParams

    with open(os.environ["TAILGUARDKV_VLLM_PROFILE_PAYLOAD_PATH"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    llm = LLM(
        model=payload["model"],
        download_dir=payload.get("cache_dir") or None,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=bool(payload.get("enforce_eager", True)),
        gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.75)),
        max_model_len=int(payload.get("max_model_len", 1024)),
        tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
    )
    sampling = SamplingParams(max_tokens=int(payload["max_new_tokens"]), temperature=0.0)
    results = []
    max_new_tokens = int(payload["max_new_tokens"])
    for request in payload.get("requests", []):
        started = time.perf_counter()
        outputs = llm.generate([request["prompt"]], sampling)
        finished = time.perf_counter()
        latency_ms = (finished - started) * 1000
        output = outputs[0]
        text = output.outputs[0].text if output.outputs else ""
        metrics = getattr(output, "metrics", None)
        ttft_ms, ttft_semantics = _ttft_from_metrics(
            metrics,
            max_new_tokens=max_new_tokens,
            request_start=started,
            request_end=finished,
        )
        resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        results.append(
            {
                "request_id": request["request_id"],
                "ok": True,
                "output_text": text,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "ttft_semantics": ttft_semantics,
                "peak_memory_mib": resident,
                "kv_cache_memory_mib": 0.0,
                "resident_memory_mib": resident,
            }
        )
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))
    return 0


def _ttft_from_metrics(metrics, *, max_new_tokens: int, request_start=None, request_end=None):
    if metrics is None and max_new_tokens == 1 and request_start is not None and request_end is not None:
        return max(0.0, (float(request_end) - float(request_start)) * 1000), "first_token"
    if metrics is None:
        return None, "unavailable"
    arrival_time = getattr(metrics, "arrival_time", None)
    first_token_time = getattr(metrics, "first_token_time", None)
    if first_token_time is None and max_new_tokens == 1:
        first_token_time = getattr(metrics, "last_token_time", None)
    if first_token_time is None or arrival_time is None:
        return None, "unavailable"
    return max(0.0, (float(first_token_time) - float(arrival_time)) * 1000), "first_token"


if __name__ == "__main__":
    raise SystemExit(main())
