#!/usr/bin/env python3
"""Check TailGuardKV conda environment connectivity.

Run from the host shell:
    conda run -n tailguardkv-base python scripts/check_envs.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvCheck:
    role: str
    env: str
    modules: tuple[str, ...]
    pythonpath: tuple[str, ...] = ()


DEFAULT_CHECKS = (
    EnvCheck(
        "base",
        "tailguardkv-base",
        (
            "torch",
            "transformers",
            "pandas",
            "numpy",
            "sklearn",
            "lightgbm",
            "matplotlib",
            "pyarrow",
        ),
    ),
    EnvCheck("eng_vllm", "edgekv-vllm0110", ("torch", "vllm", "transformers", "ray")),
    EnvCheck("eng_lmcache", "h3-lmcache-blog", ("torch", "vllm", "lmcache", "transformers")),
    EnvCheck("comp_kivi", "edgekv-kivi", ("torch", "transformers", "models", "quant", "kivi_gemv")),
    EnvCheck(
        "comp_h2o_snapkv",
        "edgekv-h2o",
        ("torch", "transformers", "utils_hh.modify_llama", "snapkv.monkeypatch.snapkv_utils"),
        ("third_party/H2O/h2o_hf",),
    ),
)


PROBE = r"""
import importlib
import json
import sys

result = {"python": sys.version.split()[0], "modules": {}}
for name in __MODULES__:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {
            "ok": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except Exception as exc:
        result["modules"][name] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc)[:300],
        }
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
"""


def run_check(check: EnvCheck, timeout: int) -> dict[str, object]:
    code = PROBE.replace("__MODULES__", repr(list(check.modules)))
    env = os.environ.copy()
    if check.pythonpath:
        paths = [os.path.abspath(path) for path in check.pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
    started = time.monotonic()
    proc = subprocess.run(
        ["conda", "run", "-n", check.env, "python", "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    entry: dict[str, object] = {
        "role": check.role,
        "env": check.env,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        entry["ok"] = False
        entry["error"] = (proc.stderr or proc.stdout).strip()[-1000:]
        return entry
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        entry["ok"] = False
        entry["error"] = f"failed to parse probe output: {exc}; output={proc.stdout[-1000:]}"
        return entry
    entry.update(payload)
    entry["ok"] = all(mod["ok"] for mod in payload["modules"].values())
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [run_check(check, args.timeout) for check in DEFAULT_CHECKS]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "OK" if result["ok"] else "FAIL"
            print(f"[{status}] {result['role']} env={result['env']} elapsed={result['elapsed_s']}s")
            if "modules" in result:
                for name, info in result["modules"].items():
                    if info["ok"]:
                        print(f"  - {name}: {info['version']}")
                    else:
                        print(f"  - {name}: {info['error']}")
            elif "error" in result:
                print(f"  - {result['error']}")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
