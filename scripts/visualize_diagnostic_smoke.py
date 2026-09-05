#!/usr/bin/env python3
"""Render a diagnostic-only completion chart from a profile measurement CSV."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    completed = Counter()
    with Path(args.profiles).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("ok")).lower() == "true" and str(row.get("measured")).lower() == "true":
                completed[str(row.get("profile") or "unknown")] += 1
    failures = Counter()
    with Path(args.failed).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            error = str(row.get("error") or "unknown failure").splitlines()[0]
            failures[error] += 1

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = list(completed) or ["no completed profile"]
    values = [completed[name] for name in names] or [0]
    axes[0].bar(names, values, color="#37718e")
    axes[0].axhline(args.expected_requests, color="#d1495b", linestyle="--", label="expected requests")
    axes[0].set_title("Diagnostic-only profile completion")
    axes[0].set_ylabel("Measured requests")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    labels = list(failures) or ["no failures"]
    axes[1].barh(labels, [failures[label] for label in labels], color="#d1495b")
    axes[1].set_title("Termination reason")
    axes[1].set_xlabel("Failed chunks")
    axes[1].tick_params(axis="y", labelsize=8)
    figure.suptitle("Diagnostic-only: not a formal baseline comparison", fontsize=12)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
