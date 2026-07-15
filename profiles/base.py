from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence

from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult


class ProfileAdapter(ABC):
    """统一封装 full、量化、剪枝等 profile 的最小接口。"""

    name: str
    env: str

    @abstractmethod
    def profiles(self) -> tuple[ProfileSpec, ...]:
        ...

    @abstractmethod
    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ...

    @abstractmethod
    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        ...

    def profile_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.profiles())

    def get_profile(self, profile_name: str) -> ProfileSpec:
        for spec in self.profiles():
            if spec.name == profile_name:
                return spec
        raise KeyError(f"{self.name} 没有 profile: {profile_name}")


def run_conda_probe(
    env_name: str,
    modules: Sequence[str],
    timeout_s: int = 120,
    pythonpath: Sequence[str] = (),
) -> tuple[bool, dict[str, str], str | None]:
    code = """
import importlib
import json
import sys

payload = {"python": sys.version.split()[0], "modules": {}}
for name in __MODULES__:
    try:
        module = importlib.import_module(name)
        payload["modules"][name] = {
            "ok": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except Exception as exc:
        payload["modules"][name] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc)[:300],
        }
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
""".replace("__MODULES__", repr(list(modules)))

    env = os.environ.copy()
    if pythonpath:
        paths = [os.path.abspath(path) for path in pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)

    proc = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        return False, {}, (proc.stderr or proc.stdout).strip()[-1000:]
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return False, {}, f"无法解析探测输出: {exc}; output={proc.stdout[-1000:]}"

    versions = {"python": str(payload["python"])}
    ok = True
    for module_name, info in payload["modules"].items():
        versions[module_name] = str(info.get("version") if info["ok"] else info.get("error"))
        ok = ok and bool(info["ok"])
    return ok, versions, None if ok else "至少一个模块导入失败"


def dry_profile_measurement(
    adapter: str,
    request: Request,
    spec: ProfileSpec,
    latency_ms: float,
    peak_memory_mib: float,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=True,
        measured=False,
        output_text=request.prompt,
        latency_ms=latency_ms,
        ttft_ms=latency_ms,
        peak_memory_mib=peak_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=0.0 if spec.exact else None,
        extra={
            "family": spec.family,
            "note": "dry_run仅验证统一表结构，尚未执行真实模型和profile kernel",
        },
    )
