"""Chart generators for FloodCut dashboard using Matplotlib.

All charts output clear titles, axes, legends, and explicit provenance tags.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from floodcut.failure import FailureSimSummary
from floodcut.maxflow import FloodCutResult
from floodcut.rainfall_loader import MONTH_NAMES, MONSOON_MONTHS, RainfallDataset
from floodcut.repair import OptimisationReport


def plot_rainfall_trends(ds: RainfallDataset, year_range: tuple[int, int]) -> plt.Figure:
    """Plot annual and monsoon rainfall totals across years."""
    records = [r for r in ds.records if year_range[0] <= r.year <= year_range[1]]
    years = [r.year for r in records]
    annual_totals = [r.annual_total_mm() or 0.0 for r in records]
    monsoon_totals = [r.monsoon_total_mm() or 0.0 for r in records]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)

    ax.plot(years, annual_totals, label="Annual Total (mm)", color="#1976d2", linewidth=1.5)
    ax.plot(years, monsoon_totals, label="Monsoon Total (Jun–Sep mm)", color="#388e3c", linewidth=2.0)

    if monsoon_totals:
        avg_monsoon = np.mean([m for m in monsoon_totals if m > 0])
        ax.axhline(
            avg_monsoon,
            color="#d32f2f",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean Monsoon: {avg_monsoon:.0f} mm",
        )

    ax.set_title(
        f"Historical Rainfall Trend ({year_range[0]}–{year_range[1]}) "
        f"[{ds.dataset_record.provenance}]",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Accumulated Rainfall (mm)", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_monthly_distribution(ds: RainfallDataset, selected_year: int) -> plt.Figure:
    """Plot monthly accumulated rainfall for a chosen year."""
    rec = ds.record_for_year(selected_year)
    values = [rec.monthly_mm.get(m) or 0.0 if rec else 0.0 for m in MONTH_NAMES]

    colors = [
        "#2e7d32" if m in MONSOON_MONTHS else "#90caf9" for m in MONTH_NAMES
    ]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    bars = ax.bar(MONTH_NAMES, values, color=colors, edgecolor="#333", linewidth=0.7)

    # Highlight peak month
    if values and max(values) > 0:
        max_idx = int(np.argmax(values))
        bars[max_idx].set_edgecolor("#b71c1c")
        bars[max_idx].set_linewidth(2.0)

    ax.set_title(
        f"Monthly Rainfall in {selected_year} "
        f"(Total: {rec.annual_total_mm() if rec else 0.0:.1f} mm) "
        f"[{ds.dataset_record.provenance}]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Monthly Accumulated (mm)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Note
    ax.text(
        0.02,
        0.92,
        "Green: Monsoon (Jun–Sep)\nBlue: Non-monsoon",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    return fig


def plot_flow_comparison(
    baseline_result: FloodCutResult,
    modified_result: FloodCutResult,
    modified_label: str = "Failure Scenario",
) -> plt.Figure:
    """Compare baseline vs modified flow network metrics (max flow and overflow)."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    categories = ["Demand", "Deliverable Flow", "Overflow (Shortfall)"]
    base_vals = [
        baseline_result.total_demand_m3s,
        baseline_result.max_flow_m3s,
        baseline_result.total_overflow_m3s,
    ]
    mod_vals = [
        modified_result.total_demand_m3s,
        modified_result.max_flow_m3s,
        modified_result.total_overflow_m3s,
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, base_vals, width, label="Baseline Network", color="#0275d8")
    ax.bar(x + width / 2, mod_vals, width, label=modified_label, color="#d9534f")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Discharge rate (m³/s)", fontsize=10)
    ax.set_title("Flow Capacity & Overflow Comparison [SYNTHETIC Network]", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate delta overflow
    delta_ov = modified_result.total_overflow_m3s - baseline_result.total_overflow_m3s
    sign = "+" if delta_ov >= 0 else ""
    ax.text(
        x[2],
        max(base_vals[2], mod_vals[2]) + 0.15,
        f"Δ Overflow: {sign}{delta_ov:.3f} m³/s",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#b71c1c" if delta_ov > 0 else "#2e7d32",
    )

    fig.tight_layout()
    return fig


def plot_strategy_comparison(report: OptimisationReport) -> plt.Figure:
    """Compare repair strategies: cost vs overflow reduction."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)

    strat_names = list(report.plans.keys())
    reductions = [report.plans[s].overflow_reduction_m3s for s in strat_names]
    costs = [report.plans[s].total_cost for s in strat_names]

    x = np.arange(len(strat_names))
    width = 0.38

    ax.bar(x - width / 2, reductions, width, label="Overflow Reduction (m³/s)", color="#2e7d32")
    ax.bar(x + width / 2, costs, width, label="Repair Cost (Units)", color="#f57c00")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strat_names], fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_title(
        f"Repair Optimization Strategy Comparison (Budget = {report.budget:.1f} Units)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Highlight optimal if exhaustive ran
    if report.exhaustive_optimal is not None:
        ax.axhline(
            report.exhaustive_optimal.overflow_reduction_m3s,
            color="#1b5e20",
            linestyle="--",
            alpha=0.7,
            label=f"Max Possible Reduction: {report.exhaustive_optimal.overflow_reduction_m3s:.2f} m³/s",
        )

    fig.tight_layout()
    return fig
