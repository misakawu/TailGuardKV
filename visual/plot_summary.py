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
    ("p95_ttft_ms", "summary_policy_p95_ttft.png", "Backend P95 TTFT by budget and constraint", "Backend P95 TTFT (ms)"),
    ("mean_kv_cache_memory_mib", "summary_policy_kv_memory.png", "Backend KV cache outcome by budget and constraint", "Backend mean KV cache memory (MiB)"),
    ("p95_quality_loss", "summary_policy_quality_loss.png", "Backend quality outcome by budget and constraint", "Backend P95 quality loss"),
    ("violation_rate", "summary_policy_violation_rate.png", "Backend violation rate by budget and constraint", "Backend violation rate"),
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


def _constraint_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        epsilon = row.get("epsilon") or ""
        delta = row.get("delta") or ""
        grouped[(epsilon, delta)].append(row)
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (
                _numeric(item[0][0]) if _numeric(item[0][0]) is not None else math.inf,
                _numeric(item[0][1]) if _numeric(item[0][1]) is not None else math.inf,
            ),
        )
    )


def _line_chart(rows: list[dict[str, str]], metric: str, output: Path, title: str, ylabel: str) -> Path | None:
    grouped = _constraint_groups(rows)
    panels: list[tuple[tuple[str, str], list[str], dict[str, list[float | None]]]] = []
    for constraint, group_rows in grouped.items():
        labels, series = _policy_metric_series(group_rows, metric)
        if labels and series:
            panels.append((constraint, labels, series))
    if not panels:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = 2 if len(panels) > 1 else 1
    rows_count = math.ceil(len(panels) / columns)
    max_labels = max(len(labels) for _, labels, _ in panels)
    width = max(7.0, min(18.0, 1.1 * max_labels + 4.0 * columns))
    height = max(4.5, 4.2 * rows_count)
    fig, axes = plt.subplots(rows_count, columns, figsize=(width, height), squeeze=False)
    flat_axes = [axis for row_axes in axes for axis in row_axes]
    for axis, (constraint, labels, series) in zip(flat_axes, panels, strict=False):
        x_values = list(range(len(labels)))
        for policy, values in series.items():
            if all(value is None for value in values):
                continue
            axis.plot(x_values, values, marker="o", linewidth=1.7, markersize=4.0, label=policy)
        epsilon, delta = constraint
        axis.set_title(f"epsilon={epsilon or '?'} delta={delta or '?'}")
        axis.set_xlabel("Memory budget")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels)
        axis.tick_params(axis="x", rotation=0, labelsize=8)
        axis.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    fig.suptitle(title)
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
        chart = _line_chart(rows, metric, destination / filename, title, ylabel)
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
