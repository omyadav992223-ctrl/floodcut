"""Automated tests for Phase 6: Frontend and Visualization.

Verifies:
1. UI state and data loader caches.
2. Folium map generation and custom HTML legend.
3. Offline Matplotlib network graph fallback.
4. Rainfall trend and monthly distribution charts.
5. Flow comparison and repair strategy comparison charts.
6. Backend integration for failure simulations and repair optimization.
7. CLI entry point for dashboard argument parsing.
8. Provenance audit integrity across frontend helpers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from floodcut.failure import _apply_failures, enumerate_failures
from floodcut.graph import build_flow_network
from floodcut.maxflow import analyze
from floodcut.models import DrainageNetwork, RainfallScenario
from floodcut.provenance import REAL, SYNTHETIC, base_category
from floodcut.rainfall_loader import load_synthetic_fixture
from floodcut.repair import compare_strategies
from floodcut.ui.charts import (
    plot_flow_comparison,
    plot_monthly_distribution,
    plot_rainfall_trends,
    plot_strategy_comparison,
)
from floodcut.ui.map_view import build_mumbai_folium_map, plot_offline_drainage_graph
from floodcut.ui.state import (
    get_drainage_network,
    get_hotspot_layer,
    get_imd_grid_layer,
    get_rainfall_dataset,
    get_scenarios,
)


# ---------------------------------------------------------------------------
# 1. State and Data Loader Tests
# ---------------------------------------------------------------------------

class TestUIStateLoaders:
    def test_get_drainage_network_returns_valid_network(self):
        net = get_drainage_network()
        assert isinstance(net, DrainageNetwork)
        assert len(net.junctions) == 14
        assert len(net.drains) == 14
        assert net.metadata.is_simulated is True

    def test_get_scenarios_returns_standard_scenarios(self):
        scenarios = get_scenarios()
        assert isinstance(scenarios, dict)
        assert "design" in scenarios
        assert "extreme" in scenarios
        assert scenarios["design"].intensity_mm_h == 50.0

    def test_get_rainfall_dataset_returns_dataset_and_bool(self):
        ds, is_real = get_rainfall_dataset()
        assert ds is not None
        assert len(ds.records) > 0
        assert isinstance(is_real, bool)

    def test_get_imd_grid_layer_has_provenance(self):
        layer = get_imd_grid_layer()
        assert len(layer.points) >= 1
        assert layer.points[0].provenance == "REAL:IMD_GRID"

    def test_get_hotspot_layer_has_provenance(self):
        layer = get_hotspot_layer()
        assert len(layer.points) >= 5
        assert all(p.provenance == "DERIVED:GEOCODED" for p in layer.points)


# ---------------------------------------------------------------------------
# 2. Map and Graph Generation Tests
# ---------------------------------------------------------------------------

class TestMapView:
    def test_build_mumbai_folium_map(self):
        net = get_drainage_network()
        imd_grid = get_imd_grid_layer()
        hotspots = get_hotspot_layer()

        m = build_mumbai_folium_map(
            net,
            imd_grid,
            hotspots,
            show_drains=True,
            show_hotspots=True,
            show_imd_grid=True,
        )
        assert m is not None
        html_str = m.get_root().render()
        assert "FloodCut Map Legend" in html_str
        assert "REAL" in html_str
        assert "SYNTHETIC" in html_str
        assert "DERIVED" in html_str

    def test_build_map_layer_toggles(self):
        net = get_drainage_network()
        imd_grid = get_imd_grid_layer()
        hotspots = get_hotspot_layer()

        # Should render even with all layers toggled off
        m = build_mumbai_folium_map(
            net,
            imd_grid,
            hotspots,
            show_drains=False,
            show_hotspots=False,
            show_imd_grid=False,
        )
        assert m is not None

    def test_plot_offline_drainage_graph(self):
        net = get_drainage_network()
        fig = plot_offline_drainage_graph(net)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Chart Generation Tests
# ---------------------------------------------------------------------------

class TestUICharts:
    def test_plot_rainfall_trends(self):
        ds = load_synthetic_fixture()
        fig = plot_rainfall_trends(ds, (2005, 2007))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_monthly_distribution(self):
        ds = load_synthetic_fixture()
        fig = plot_monthly_distribution(ds, 2005)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_flow_comparison(self):
        net = get_drainage_network()
        sc = get_scenarios()["design"]
        fn = build_flow_network(net, sc)
        base_res = analyze(net, fn)

        # Failure run
        failed_net = _apply_failures(net, ["D06"])
        fn_fail = build_flow_network(failed_net, sc)
        fail_res = analyze(failed_net, fn_fail)

        fig = plot_flow_comparison(base_res, fail_res, modified_label="Failure D06")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_strategy_comparison(self):
        net = get_drainage_network()
        sc = get_scenarios()["design"]
        report = compare_strategies(net, sc, budget=25.0, run_exhaustive=True)
        fig = plot_strategy_comparison(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Backend-to-Frontend Algorithm Integration
# ---------------------------------------------------------------------------

class TestAlgorithmIntegration:
    def test_failure_simulation_execution(self):
        net = get_drainage_network()
        sc = get_scenarios()["design"]
        base_fn = build_flow_network(net, sc)
        base_res = analyze(net, base_fn)

        # Block D08a
        fail_net = _apply_failures(net, ["D08a"])
        fail_fn = build_flow_network(fail_net, sc)
        fail_res = analyze(fail_net, fail_fn)

        assert fail_res.total_overflow_m3s > base_res.total_overflow_m3s
        assert fail_res.max_flow_m3s < base_res.max_flow_m3s

    def test_worst_case_k_enumeration(self):
        net = get_drainage_network()
        sc = get_scenarios()["design"]
        summary = enumerate_failures(net, sc, k_max=1)
        wc = summary.worst_case(1)
        assert wc is not None
        assert wc.total_overflow_m3s >= summary.base_overflow_m3s

    def test_repair_optimization_execution(self):
        net = get_drainage_network()
        sc = get_scenarios()["design"]
        report = compare_strategies(net, sc, budget=30.0, run_exhaustive=True)
        assert "greedy" in report.plans
        assert "exhaustive" in report.plans
        assert report.plans["greedy"].overflow_reduction_m3s >= 0.0


# ---------------------------------------------------------------------------
# 5. CLI Integration
# ---------------------------------------------------------------------------

class TestCLIAppIntegration:
    def test_cli_app_parser_help(self):
        from floodcut.cli import main
        # Trigger help flag check
        with pytest.raises(SystemExit) as excinfo:
            main(["app", "--help"])
        assert excinfo.value.code == 0

    def test_cli_dashboard_parser_help(self):
        from floodcut.cli import main
        with pytest.raises(SystemExit) as excinfo:
            main(["dashboard", "--help"])
        assert excinfo.value.code == 0
