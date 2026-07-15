from __future__ import annotations


def mib_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)
