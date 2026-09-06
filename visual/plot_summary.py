from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_util.session_aggregation import bootstrap_ci_columns


ChartSpec = tuple[str, str, str, str]


POLICY_CHARTS: tuple[ChartSpec, ...] = (
    ("p95_ttft_ms", "summary_policy_p95_ttft.png", "Server-side replay P95 TTFT by budget and constraint", "Replay outcome P95 TTFT (ms)"),
    ("mean_kv_cache_memory_mib", "summary_policy_kv_memory.png", "Server-side replay KV residency by budget and constraint", "Replay outcome mean resident KV (MiB)"),
    ("p95_quality_loss", "summary_policy_quality_loss.png", "Server-side replay quality by budget and constraint", "Replay outcome P95 quality loss"),
    ("violation_rate", "summary_policy_violation_rate.png", "Server-side replay violation rate by budget and constraint", "Replay violation rate"),
)
RISK_ANNOTATED_CHARTS = {"p95_quality_loss", "violation_rate"}
RISK_LABEL = "diagnostic_only \u00b7 risk_evidence_insufficient"


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


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
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


def _numeric_pair(row: dict[str, str], low_key: str, high_key: str) -> tuple[float | None, float | None]:
    low = _numeric(row.get(low_key))
    high = _numeric(row.get(high_key))
    if low is None or high is None:
        return None, None
    return low, high


def _is_diagnostic_risk(row: dict[str, str]) -> bool:
    return str(row.get("quality_status") or "").strip() == "risk_evidence_insufficient"


def _policy_metric_series(
    rows: list[dict[str, str]],
    metric: str,
) -> tuple[list[str], dict[str, list[float | None]], dict[str, dict[str, tuple[float, float]]]]:
    cell_rows: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, float]] = defaultdict(dict)
    ci_low_key, ci_high_key = bootstrap_ci_columns(metric)
    bands: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
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
        low, high = _numeric_pair(row, ci_low_key, ci_high_key)
        if low is not None and high is not None:
            bands[policy][label] = (low, high)
    labels = sorted(cell_rows, key=lambda label: _cell_sort_key(cell_rows[label]))
    series = {
        policy: [policy_values.get(label) for label in labels]
        for policy, policy_values in sorted(values.items())
    }
    return labels, series, {policy: dict(band_values) for policy, band_values in bands.items()}


def _policy_session_values(
    session_rows: list[dict[str, str]],
    metric: str,
    cell_key: tuple[str, str],
) -> dict[tuple[str, str], list[float]]:
    if metric not in SESSION_POINT_AVAILABLE:
        return {}
    points: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in session_rows:
        epsilon = str(row.get("epsilon") or "")
        delta = str(row.get("delta") or "")
        if (epsilon, delta) != cell_key:
            continue
        policy = str(row.get("policy") or "")
        label = _cell_label(row)
        if not policy or not label:
            continue
        value = _numeric(row.get(metric))
        if value is not None:
            points[(policy, label)].append(value)
    return dict(points)


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


SESSION_POINT_AVAILABLE = {
    "p95_ttft_ms",
    "mean_ttft_ms",
    "mean_kv_cache_memory_mib",
    "p95_quality_loss",
    "mean_quality_loss",
    "violation_rate",
}


def _line_chart(
    rows: list[dict[str, str]],
    metric: str,
    output: Path,
    title: str,
    ylabel: str,
    session_rows: list[dict[str, str]] | None = None,
) -> Path | None:
    grouped = _constraint_groups(rows)
    panels: list[tuple[tuple[str, str], list[str], dict[str, list[float | None]], dict[str, dict[str, tuple[float, float]]]]] = []
    for constraint, group_rows in grouped.items():
        labels, series, bands = _policy_metric_series(group_rows, metric)
        if labels and series:
            panels.append((constraint, labels, series, bands))
    if not panels:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = 2 if len(panels) > 1 else 1
    rows_count = math.ceil(len(panels) / columns)
    max_labels = max(len(labels) for _, labels, _, _ in panels)
    width = max(7.0, min(18.0, 1.1 * max_labels + 4.0 * columns))
    height = max(4.5, 4.2 * rows_count)
    fig, axes = plt.subplots(rows_count, columns, figsize=(width, height), squeeze=False)
    flat_axes = [axis for row_axes in axes for axis in row_axes]
    for axis, (constraint, labels, series, bands) in zip(flat_axes, panels, strict=False):
        x_values = list(range(len(labels)))
        risk_group = group_rows_any(constraint, rows)
        session_points = (
            _policy_session_values(session_rows, metric, constraint) if session_rows is not None else {}
        )
        for policy, values in series.items():
            if all(value is None for value in values):
                continue
            axis.plot(x_values, values, marker="o", linewidth=1.7, markersize=4.0, label=policy)
            band_values = bands.get(policy, {})
            lows = [band_values[label][0] if label in band_values else None for label in labels]
            highs = [band_values[label][1] if label in band_values else None for label in labels]
            finite = [index for index, (low, high) in enumerate(zip(lows, highs, strict=True)) if low is not None and high is not None]
            if finite:
                axis.fill_between(
                    [x_values[index] for index in finite],
                    [lows[index] for index in finite],
                    [highs[index] for index in finite],
                    alpha=0.15,
                    linewidth=0,
                )
        session_values = session_points
        if session_values:
            for policy in sorted({key[0] for key in session_values}):
                x_list: list[float] = []
                y_list: list[float] = []
                for label_index, label in enumerate(labels):
                    for value in session_values.get((policy, label), []):
                        x_list.append(x_values[label_index])
                        y_list.append(value)
                if x_list:
                    axis.scatter(
                        x_list,
                        y_list,
                        marker=".",
                        s=14,
                        alpha=0.35,
                        color="0.35",
                        zorder=1.5,
                    )
        epsilon, delta = constraint
        axis.set_title(f"epsilon={epsilon or '?'} delta={delta or '?'}")
        axis.set_xlabel("Memory budget")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels)
        axis.tick_params(axis="x", rotation=0, labelsize=8)
        axis.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
        if metric in RISK_ANNOTATED_CHARTS and risk_group:
            axis.text(
                0.98,
                0.97,
                RISK_LABEL,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#8a6d3b",
            )
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def group_rows_any(constraint: tuple[str, str], rows: list[dict[str, str]]) -> bool:
    epsilon, delta = constraint
    return any(
        str(row.get("epsilon") or "") == epsilon
        and str(row.get("delta") or "") == delta
        and _is_diagnostic_risk(row)
        for row in rows
    )


def _placeholder_chart(output: Path, title: str, ylabel: str, *, risk_label: bool) -> None:
    """Write a chart file when a metric has no plottable rows (diagnostic runs)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.axis("off")
    lines = [f"{title}\n{ylabel}: no rows in this diagnostic run"]
    if risk_label:
        lines.append(RISK_LABEL)
    axis.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=10,
        transform=axis.transAxes,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_summary(
    summary_csv: str | Path,
    output_dir: str | Path | None = None,
    session_points_csv: str | Path | None = None,
    *,
    always_emit_all: bool = False,
) -> list[Path]:
    summary_path = Path(summary_csv)
    destination = Path(output_dir) if output_dir is not None else summary_path.parent
    rows = _read_rows(summary_path)
    session_rows = _read_rows(Path(session_points_csv)) if session_points_csv is not None else []
    outputs: list[Path] = []
    for metric, filename, title, ylabel in POLICY_CHARTS:
        chart = _line_chart(rows, metric, destination / filename, title, ylabel, session_rows=session_rows)
        if chart is not None:
            outputs.append(chart)
            continue
        if not always_emit_all:
            continue
        placeholder = destination / filename
        _placeholder_chart(placeholder, title, ylabel, risk_label=metric in RISK_ANNOTATED_CHARTS)
        outputs.append(placeholder)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TailGuardKV summary PNG charts.")
    parser.add_argument("summary_csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--session-points-csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = plot_summary(args.summary_csv, args.output_dir, getattr(args, "session_points_csv", None))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
