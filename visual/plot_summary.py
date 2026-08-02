from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ChartSpec = tuple[str, str, str, str]


POLICY_CHARTS: tuple[ChartSpec, ...] = (
    ("p95_ttft_ms", "summary_policy_p95_ttft.png", "P95 TTFT by budget and constraint", "P95 TTFT (ms)"),
    ("mean_kv_cache_memory_mib", "summary_policy_kv_memory.png", "KV cache memory by budget and constraint", "Mean KV cache memory (MiB)"),
    ("p95_quality_loss", "summary_policy_quality_loss.png", "Quality loss by budget and constraint", "P95 quality loss"),
    ("violation_rate", "summary_policy_violation_rate.png", "Violation rate by budget and constraint", "Violation rate"),
)


def _numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _read_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _policy_name(row: dict[str, str]) -> str:
    if "section" in row and row.get("section") != "policy":
        return ""
    return row.get("policy") or row.get("name") or ""


def _cell_sort_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    memory = _numeric(row.get("memory_budget_mib"))
    epsilon = _numeric(row.get("epsilon"))
    delta = _numeric(row.get("delta"))
    label = _cell_label(row)
    return (
        memory if memory is not None else math.inf,
        epsilon if epsilon is not None else math.inf,
        delta if delta is not None else math.inf,
        label,
    )


def _format_number(value: str | None) -> str:
    number = _numeric(value)
    if number is None:
        return str(value or "")
    return f"{number:g}"


def _cell_label(row: dict[str, str]) -> str:
    budget = _format_number(row.get("memory_budget_mib"))
    epsilon = _format_number(row.get("epsilon"))
    delta = _format_number(row.get("delta"))
    parts = []
    if budget:
        parts.append(f"B={budget}")
    if epsilon:
        parts.append(f"e={epsilon}")
    if delta:
        parts.append(f"d={delta}")
    return "\n".join(parts)


def _policy_metric_series(rows: list[dict[str, str]], metric: str) -> tuple[list[str], dict[str, list[float | None]]]:
    cell_rows: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        policy = _policy_name(row)
        if not policy:
            continue
        value = _numeric(row.get(metric))
        if value is None:
            continue
        label = _cell_label(row)
        if not label:
            continue
        cell_rows.setdefault(label, row)
        values[policy][label] = value
    labels = sorted(cell_rows, key=lambda label: _cell_sort_key(cell_rows[label]))
    series = {
        policy: [policy_values.get(label) for label in labels]
        for policy, policy_values in sorted(values.items())
    }
    return labels, series


def _line_chart(labels: list[str], series: dict[str, list[float | None]], output: Path, title: str, ylabel: str) -> Path | None:
    if not labels or not series:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    width = max(7.0, min(18.0, 1.1 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    x_values = list(range(len(labels)))
    for policy, values in series.items():
        if all(value is None for value in values):
            continue
        ax.plot(x_values, values, marker="o", linewidth=1.7, markersize=4.0, label=policy)
    ax.set_title(title)
    ax.set_xlabel("Memory budget x quality constraint")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_summary(summary_csv: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    summary_path = Path(summary_csv)
    destination = Path(output_dir) if output_dir is not None else summary_path.parent
    rows = _read_rows(summary_path)
    outputs: list[Path] = []
    for metric, filename, title, ylabel in POLICY_CHARTS:
        labels, series = _policy_metric_series(rows, metric)
        chart = _line_chart(labels, series, destination / filename, title, ylabel)
        if chart is not None:
            outputs.append(chart)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TailGuardKV summary PNG charts.")
    parser.add_argument("summary_csv")
    parser.add_argument("--output-dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = plot_summary(args.summary_csv, args.output_dir)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
