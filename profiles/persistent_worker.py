from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from typing import Any


def _response_error(message: str, *, fatal: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": message,
        "worker": {"mode": "persistent"},
    }
    if fatal:
        payload["fatal_error"] = message
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent profile worker server.")
    parser.add_argument("--runtime-module", required=True)
    args = parser.parse_args()

    runtime_module = importlib.import_module(args.runtime_module)
    worker_state: dict[str, Any] = {}
    exit_code = 0

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except Exception as exc:
            response = _response_error(f"invalid worker message: {type(exc).__name__}: {exc}", fatal=True)
            print(json.dumps(response, ensure_ascii=False), flush=True)
            return 1

        op = str(message.get("op") or "")
        try:
            if op == "init":
                response = runtime_module.worker_init(message, worker_state)
            elif op == "run_batch":
                response = runtime_module.worker_run_batch(message, worker_state)
            elif op == "shutdown":
                response = runtime_module.worker_shutdown(message, worker_state)
                print(json.dumps(response, ensure_ascii=False), flush=True)
                return 0 if response.get("ok") else 1
            else:
                response = _response_error(f"unsupported worker op: {op}", fatal=True)
                exit_code = 1
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}"
            response = _response_error(detail, fatal=True)
            exit_code = 1

        print(json.dumps(response, ensure_ascii=False), flush=True)
        if response.get("fatal_error"):
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
