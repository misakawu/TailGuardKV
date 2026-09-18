# Disable Summary CI Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove confidence-interval shading from all four policy summary PNG charts while retaining CI data in the CSV outputs.

**Architecture:** `plot_summary` continues to read policy values and render the existing line series. CI fields remain available to aggregation consumers, but the plotting path no longer builds CI bands or calls Matplotlib `fill_between`.

**Tech Stack:** Python 3, pytest, CSV, Matplotlib.

## Global Constraints

- Do not modify bootstrap calculations or total-summary CSV columns.
- Preserve chart names, line-series values, TTFT y-axis cap, and risk labels.
- Regenerate only the four existing summary PNG files from `session27_total_summary.csv`; do not run GPU experiments.

---

### Task 1: Remove CI bands from summary chart rendering

**Files:**
- Modify: `tests/test_session_summary_visuals.py`
- Modify: `visual/plot_summary.py`

**Interfaces:**
- Consumes: `plot_summary(summary_csv, output_dir, session_points_csv=None)`.
- Produces: the same four PNG paths, with `Axes.collections` empty when the input only supplies CI bands.

- [ ] **Step 1: Write the failing test**

```python
def test_plot_summary_does_not_render_ci_bands(tmp_path: Path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    collections: list[int] = []
    original_savefig = Figure.savefig
    def capture_savefig(self, *args, **kwargs):
        collections.extend(len(axis.collections) for axis in self.axes)
        return original_savefig(self, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    plot_summary(summary_csv, tmp_path)
    assert collections and all(count == 0 for count in collections)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/test_session_summary_visuals.py::test_plot_summary_does_not_render_ci_bands -v`
Expected: FAIL because the current renderer calls `fill_between` for each policy CI.

- [ ] **Step 3: Write the minimal implementation**

Remove the CI-band return value from `_policy_metric_series` and remove the `axis.fill_between(...)` branch in `_line_chart`. Do not alter `plot_summary` arguments or CSV aggregation.

- [ ] **Step 4: Run focused visual tests**

Run: `pytest tests/test_session_summary_visuals.py -v`
Expected: PASS with four PNG names still emitted.

### Task 2: Regenerate the requested charts

**Files:**
- Regenerate: `out/full_25session_baseline_9_14/full/policy_attempts/full40_20260916_h2o_static_safe/seed20260906/policy_tables/summary_policy_*.png`

**Interfaces:**
- Consumes: `session27_total_summary.csv`.
- Produces: the four existing summary PNG files without CI bands.

- [ ] **Step 1: Invoke the plot CLI**

```bash
python visual/plot_summary.py \
  out/full_25session_baseline_9_14/full/policy_attempts/full40_20260916_h2o_static_safe/seed20260906/policy_tables/session27_total_summary.csv \
  --output-dir out/full_25session_baseline_9_14/full/policy_attempts/full40_20260916_h2o_static_safe/seed20260906/policy_tables \
  --session-points-csv out/full_25session_baseline_9_14/full/policy_attempts/full40_20260916_h2o_static_safe/seed20260906/policy_tables/session27_session_points.csv
```

- [ ] **Step 2: Verify output artifacts**

Confirm all four `summary_policy_*.png` files exist and have modification times later than `session27_total_summary.csv`.
