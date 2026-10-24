"""FloodCut: Mumbai Flood Risk & Drainage Optimization — Phase 6 Dashboard.

Interactive frontend built with Streamlit and Folium for Discrete Mathematics
modeling, flow capacity screening, failure simulation, and repair optimization.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False

from floodcut.failure import _apply_failures, enumerate_failures
from floodcut.graph import build_flow_network
from floodcut.maxflow import analyze
from floodcut.models import RainfallScenario
from floodcut.provenance import (
    ASSUMED,
    DERIVED,
    REAL,
    SYNTHETIC,
    base_category,
)
from floodcut.rainfall_loader import (
    MONTH_NAMES,
    make_scenario_from_record,
    monthly_mm_to_intensity_mm_h,
)
from floodcut.repair import (
    _candidate_drains,
    compare_strategies,
    exhaustive_repair,
    greedy_repair,
)
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


def main() -> None:
    st.set_page_config(
        page_title="FloodCut — Mumbai Drainage Optimization",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load shared datasets
    net = get_drainage_network()
    scenarios = get_scenarios()
    rainfall_ds, is_real_rainfall = get_rainfall_dataset()
    imd_grid = get_imd_grid_layer()
    hotspots = get_hotspot_layer()

    # Sidebar Navigation
    st.sidebar.title("🌊 FloodCut")
    st.sidebar.caption("Discrete Mathematics & Drainage Optimization")
    st.sidebar.markdown("---")

    nav = st.sidebar.radio(
        "Navigation",
        [
            "Project Overview",
            "Interactive Map",
            "Rainfall Explorer",
            "Failure Simulation",
            "Repair Optimization",
            "Results & Audit",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Data Provenance Key**
        - 🟢 `REAL:IMD_GRID` (Observed)
        - 🟠 `DERIVED:GEOCODED` (Approximate)
        - 🔵 `SYNTHETIC` (Model Testbed)
        """
    )
    st.sidebar.info(
        "⚠️ **Disclaimer:** FloodCut is an academic capacity-screening model, "
        "not an operational flood forecast system."
    )

    # Route navigation
    if nav == "Project Overview":
        render_overview(net, scenarios, rainfall_ds, is_real_rainfall)
    elif nav == "Interactive Map":
        render_map(net, imd_grid, hotspots)
    elif nav == "Rainfall Explorer":
        render_rainfall_explorer(rainfall_ds, is_real_rainfall)
    elif nav == "Failure Simulation":
        render_failure_simulation(net, scenarios, rainfall_ds)
    elif nav == "Repair Optimization":
        render_repair_optimization(net, scenarios)
    elif nav == "Results & Audit":
        render_results_audit(net, scenarios)


# ---------------------------------------------------------------------------
# Section A: Project Overview
# ---------------------------------------------------------------------------

def render_overview(net, scenarios, rainfall_ds, is_real_rainfall: bool) -> None:
    st.title("🌊 FloodCut: Mumbai Flood Risk & Drainage Optimization")
    st.subheader("Discrete Mathematics Course Project — Flow Networks & Combinatorial Optimization")

    st.warning(
        "**Important Prototype Notice:** FloodCut is an academic capacity-screening prototype. "
        "It models drainage networks as capacitated directed graphs to compute maximum flow, identify minimum-cut "
        "bottlenecks, simulate combinations of drain failures, and prioritize budget-constrained repairs. "
        "The current drainage network geometry and capacities are **SYNTHETIC** illustrative fixtures and do NOT "
        "represent verified municipal underground infrastructure or real-world inundation depths."
    )

    col1, col2, col3, col4 = st.columns(4)
    counts = net.counts()
    col1.metric("Junctions (Nodes)", counts["junctions"], help="Synthetic inlet/confluence points")
    col2.metric("Drains (Edges)", counts["drains"], help="Capacitated drain conduits")
    col3.metric("Catchments", counts["catchments"], help="Rainfall demand sources")
    col4.metric("Outfalls", counts["outfalls"], help="Tidal sea and river sinks")

    st.markdown("### The Mumbai Flood-Risk Problem & Mathematical Formulation")
    st.markdown(
        """
        During extreme monsoon events, Mumbai's drainage faces compound stress from heavy rainfall intensity
        and high-tide backwater effects. FloodCut formalizes this civil infrastructure challenge into four discrete mathematical steps:

        1. **Directed Capacitated Graph Encoding:** Catchments deliver inflow via the Rational Method
           ($Q = C \\cdot i \\cdot A / 360$). Each drain is encoded with an effective capacity
           $c(e) = \\text{design\\_cap} \\times \\text{capacity\\_factor}$. Outfalls exit to super-sink $T^*$, scaled by tide.
        2. **Edmonds–Karp Max-Flow / Min-Cut Engine:** Computes maximum deliverable flow $|f|$ in polynomial time
           ($O(V \\cdot E^2)$). The residual graph partition identifies the critical bottleneck edges (the minimum cut),
           separating upstream shortfalls (simulated overflow) from delivered outfall discharge.
        3. **Combinatorial Failure Enumeration:** Evaluates all $\\binom{m}{k}$ simultaneous drain failure subsets to identify
           worst-case vulnerabilities and quantify marginal versus combined subadditive/superadditive interactions.
        4. **Budget-Constrained Repair Optimization:** Solves the knapsack-style repair prioritization under budget $B$,
           comparing greedy marginal-gain heuristics against global exhaustive combinatorial optima.
        """
    )

    st.markdown("### Data Provenance Status")
    p_cols = st.columns(3)
    with p_cols[0]:
        st.success(
            f"**Rainfall Data:** `{'REAL:IMD_GRID' if is_real_rainfall else 'SYNTHETIC'}`\n\n"
            f"- Source: India Meteorological Department (IMD) via OpenCity\n"
            f"- Span: 121 Years (1901–2021), 0.25° gridded monthly observations\n"
            f"- Records Loaded: {len(rainfall_ds.records)}"
        )
    with p_cols[1]:
        st.warning(
            "**Geospatial Hotspots:** `DERIVED:GEOCODED`\n\n"
            "- Source: Municipal Disaster Cell reports & civic bulletins\n"
            "- Approximate coordinates (±500m) for 7 chronic waterlogging points\n"
            "- Does NOT represent surveyed flood-depth polygons"
        )
    with p_cols[2]:
        st.info(
            "**Drainage Network:** `SYNTHETIC`\n\n"
            "- Hand-designed 14-conduit benchmark topology\n"
            "- Includes parallel drains, diversions, and degraded reaches\n"
            "- Preserves Phase 1–5 regression test compatibility"
        )


# ---------------------------------------------------------------------------
# Section B: Interactive Map
# ---------------------------------------------------------------------------

def render_map(net, imd_grid, hotspots) -> None:
    st.header("🗺️ Interactive Mumbai Geospatial & Network Map")
    st.markdown(
        "Explore real observational layers alongside derived hotspot points and the synthetic drainage topology. "
        "Popups provide detailed attributes and explicit provenance tags."
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        show_drains = st.checkbox("Drainage Network (SYNTHETIC)", value=True)
    with c2:
        show_hotspots = st.checkbox("Waterlogging Hotspots (DERIVED)", value=True)
    with c3:
        show_imd = st.checkbox("IMD 0.25° Grid Point (REAL)", value=True)
    with c4:
        map_mode = st.radio("Render Engine", ["Interactive Folium", "Offline Graph"], horizontal=True)

    if map_mode == "Interactive Folium":
        if HAS_STREAMLIT_FOLIUM:
            m = build_mumbai_folium_map(
                net,
                imd_grid,
                hotspots,
                show_drains=show_drains,
                show_hotspots=show_hotspots,
                show_imd_grid=show_imd,
            )
            st_folium(m, width="100%", height=550)
        else:
            st.warning("`streamlit-folium` is not available. Falling back to offline graph visualization.")
            fig = plot_offline_drainage_graph(net)
            st.pyplot(fig)
    else:
        st.info("Offline topological representation of the synthetic network junctions and drains.")
        fig = plot_offline_drainage_graph(net)
        st.pyplot(fig)

    with st.expander("Inspect Geospatial Layer Tables"):
        t1, t2 = st.tabs(["Waterlogging Hotspots (DERIVED:GEOCODED)", "IMD Grid Station (REAL:IMD_GRID)"])
        with t1:
            hotspot_data = [
                {
                    "ID": pt.id,
                    "Name": pt.name,
                    "Ward": pt.attributes.get("ward", ""),
                    "Latitude": f"{pt.lat:.4f}°N",
                    "Longitude": f"{pt.lon:.4f}°E",
                    "Notes": pt.attributes.get("notes", ""),
                    "Provenance": pt.provenance,
                }
                for pt in hotspots.points
            ]
            st.dataframe(pd.DataFrame(hotspot_data), use_container_width=True)
        with t2:
            grid_data = [
                {
                    "Station ID": pt.id,
                    "Name": pt.name,
                    "Grid Resolution": f"{pt.attributes.get('grid_resolution_deg', '0.25')}°",
                    "Latitude": f"{pt.lat:.4f}°N",
                    "Longitude": f"{pt.lon:.4f}°E",
                    "Source": pt.attributes.get("data_source", ""),
                    "Provenance": pt.provenance,
                }
                for pt in imd_grid.points
            ]
            st.dataframe(pd.DataFrame(grid_data), use_container_width=True)


# ---------------------------------------------------------------------------
# Section C: Rainfall Explorer
# ---------------------------------------------------------------------------

def render_rainfall_explorer(ds, is_real: bool) -> None:
    st.header("🌧️ Rainfall Explorer (IMD Historical Observations)")
    st.markdown(
        f"**Dataset:** `{ds.dataset_record.name}` | **Provenance:** `{ds.dataset_record.provenance}` | "
        f"**Coverage:** {ds.dataset_record.observation_period} ({ds.dataset_record.geographic_coverage})"
    )

    if is_real:
        st.success("✅ Displaying verified historical observations from the India Meteorological Department.")
    else:
        st.info("ℹ️ Using offline synthetic fixture.")

    years = ds.years()
    if not years:
        st.error("No rainfall records found.")
        return

    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        min_yr, max_yr = int(min(years)), int(max(years))
        selected_range = st.slider("Select Year Span", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr))
    with col_filter2:
        selected_year = st.selectbox("Inspect Specific Year", options=sorted(years, reverse=True), index=0)

    # Trend Chart
    st.pyplot(plot_rainfall_trends(ds, selected_range))

    # Monthly Breakdown
    st.pyplot(plot_monthly_distribution(ds, selected_year))

    # Tabular View
    with st.expander(f"View Tabular Observations ({selected_range[0]}–{selected_range[1]})"):
        rows = []
        for r in ds.records:
            if selected_range[0] <= r.year <= selected_range[1]:
                row = {"Year": r.year}
                row.update(r.monthly_mm)
                row["Annual Total (mm)"] = round(r.annual_total_mm() or 0.0, 1)
                row["Monsoon Total (mm)"] = round(r.monsoon_total_mm() or 0.0, 1)
                row["Provenance"] = r.provenance
                rows.append(row)
        df_rain = pd.DataFrame(rows)
        st.dataframe(df_rain, use_container_width=True)


# ---------------------------------------------------------------------------
# Section D: Failure Simulation
# ---------------------------------------------------------------------------

def render_failure_simulation(net, scenarios, rainfall_ds) -> None:
    st.header("⚠️ Drain Failure Simulation (Combinatorial Analysis)")
    st.markdown(
        "Evaluate network resilience by blocking one or more drains. "
        "The model recomputes maximum flow, min cut, and unmet demand shortfall."
    )

    # Scenario selection
    sc_options = list(scenarios.keys())
    c_sc1, c_sc2 = st.columns(2)
    with c_sc1:
        sc_choice = st.selectbox("Rainfall & Tide Scenario", sc_options, index=1 if "design" in sc_options else 0)
        base_scenario = scenarios[sc_choice]
    with c_sc2:
        st.markdown(
            f"**Scenario Profile:** `{base_scenario.name}`\n\n"
            f"- Rainfall Intensity: **{base_scenario.intensity_mm_h:.1f} mm/h**\n"
            f"- Outfall Tide Factor: **{base_scenario.tide_factor:.2f}**"
        )

    # Base analysis
    base_fn = build_flow_network(net, base_scenario)
    base_res = analyze(net, base_fn, cross_check=False)

    eligible_drains = sorted([d.id for d in net.drains.values() if d.capacity_factor > 0])

    st.markdown("---")
    st.subheader("Simulate Drain Blockages")
    selected_failures = st.multiselect(
        "Select Drains to Completely Block (Capacity Factor = 0.0):",
        options=eligible_drains,
        default=[],
        help="Drains already fully blocked (factor = 0.0) are omitted as they have no remaining capacity to lose.",
    )

    if selected_failures:
        failed_net = _apply_failures(net, selected_failures)
        fn_fail = build_flow_network(failed_net, base_scenario)
        fail_res = analyze(failed_net, fn_fail, cross_check=False)
        added_overflow = fail_res.total_overflow_m3s - base_res.total_overflow_m3s

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rainfall Demand", f"{base_res.total_demand_m3s:.3f} m³/s")
        m2.metric("Deliverable Flow", f"{fail_res.max_flow_m3s:.3f} m³/s", delta=f"{fail_res.max_flow_m3s - base_res.max_flow_m3s:.3f} m³/s")
        m3.metric("Simulated Overflow", f"{fail_res.total_overflow_m3s:.3f} m³/s", delta=f"{added_overflow:+.3f} m³/s", delta_color="inverse")
        m4.metric("Drains Failed", len(selected_failures))

        st.pyplot(plot_flow_comparison(base_res, fail_res, modified_label=f"Failure of {selected_failures}"))

        with st.expander("Catchment Demand & Shortfall Breakdown"):
            catchment_data = []
            for cid, c in net.catchments.items():
                demand = c.runoff_m3s(base_scenario.intensity_mm_h)
                ov = fail_res.catchment_overflow_m3s.get(cid, 0.0)
                delivered = demand - ov
                catchment_data.append({
                    "Catchment ID": cid,
                    "Name": c.name,
                    "Inflow Node": c.junction_id,
                    "Area (ha)": c.area_ha,
                    "Demand (m³/s)": round(demand, 3),
                    "Delivered Flow (m³/s)": round(delivered, 3),
                    "Overflow (Shortfall m³/s)": round(ov, 3),
                    "Shortfall %": f"{100 * ov / demand:.1f}%" if demand > 0 else "0.0%",
                })
            st.dataframe(pd.DataFrame(catchment_data), use_container_width=True)
    else:
        st.info("Select one or more drains above to compute failure impacts.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Base Rainfall Demand", f"{base_res.total_demand_m3s:.3f} m³/s")
        m2.metric("Base Deliverable Flow", f"{base_res.max_flow_m3s:.3f} m³/s")
        m3.metric("Base Overflow", f"{base_res.total_overflow_m3s:.3f} m³/s")

    # Worst-Case k-Failure Enumeration
    st.markdown("---")
    st.subheader("Automated Combinatorial Worst-Case Analysis")
    k_val = st.slider("Enumerate Combinations up to Size k:", min_value=1, max_value=min(3, len(eligible_drains)), value=2)

    if st.button("Run Combinatorial Worst-Case Enumeration", type="primary"):
        with st.spinner(f"Evaluating all k-drain failure subsets up to k={k_val}..."):
            sim_summary = enumerate_failures(net, base_scenario, k_max=k_val)

        wc_rows = []
        for k in range(1, k_val + 1):
            wc = sim_summary.worst_case(k)
            if wc:
                extra = wc.total_overflow_m3s - sim_summary.base_overflow_m3s
                wc_rows.append({
                    "Failure Size (k)": k,
                    "Total Subsets Evaluated": sim_summary.count(k),
                    "Worst Failure Subset": str(list(wc.failed_drain_ids)),
                    "Post-Failure Flow (m³/s)": round(wc.max_flow_m3s, 4),
                    "Post-Failure Overflow (m³/s)": round(wc.total_overflow_m3s, 4),
                    "Extra Overflow Added (m³/s)": round(extra, 4),
                    "Total Repair Cost": wc.total_repair_cost,
                })
        st.table(pd.DataFrame(wc_rows))


# ---------------------------------------------------------------------------
# Section E: Repair Optimization
# ---------------------------------------------------------------------------

def render_repair_optimization(net, scenarios) -> None:
    st.header("🔧 Budget-Constrained Repair Optimization")
    st.markdown(
        "Prioritize restoration of degraded drains (`capacity_factor < 1.0`) under budget constraint $B$. "
        "Compares the greedy marginal-gain heuristic with global exhaustive optimization and baseline strategies."
    )

    candidates = _candidate_drains(net)
    if not candidates:
        st.success("All drains are healthy (capacity_factor = 1.0). No repairs needed!")
        return

    # Scenario & Budget inputs
    c1, c2 = st.columns(2)
    with c1:
        sc_choice = st.selectbox("Design Scenario", list(scenarios.keys()), index=1)
        scenario = scenarios[sc_choice]
    with c2:
        total_cand_cost = sum(net.drains[d].repair_cost for d in candidates)
        budget = st.slider("Repair Budget (Cost Units):", min_value=5.0, max_value=float(total_cand_cost + 10.0), value=30.0, step=5.0)

    # Show Candidate Degraded Drains
    st.subheader(f"Degraded Drain Candidates ({len(candidates)} Drains)")
    cand_data = [
        {
            "Drain ID": did,
            "Type": net.drains[did].kind,
            "From &rarr; To": f"{net.drains[did].from_junction} &rarr; {net.drains[did].to_junction}",
            "Design Cap (m³/s)": net.drains[did].design_capacity_m3s,
            "Operating Factor": f"{net.drains[did].capacity_factor * 100:.0f}%",
            "Effective Cap (m³/s)": net.drains[did].effective_capacity_m3s,
            "Repair Cost (Units)": net.drains[did].repair_cost,
        }
        for did in candidates
    ]
    st.dataframe(pd.DataFrame(cand_data), use_container_width=True)

    # Run Optimization
    st.markdown("---")
    st.subheader("Strategy Comparison & Optimality Gap")

    report = compare_strategies(net, scenario, budget=budget, run_exhaustive=True)

    # Summary Metrics
    opt_plan = report.exhaustive_optimal
    greedy_plan = report.plans.get("greedy")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Overflow", f"{report.base_overflow_m3s:.3f} m³/s")
    if greedy_plan:
        m2.metric("Greedy Reduction", f"{greedy_plan.overflow_reduction_m3s:.3f} m³/s", delta=f"Cost: {greedy_plan.total_cost:.1f}")
    if opt_plan:
        m3.metric("Optimal Reduction", f"{opt_plan.overflow_reduction_m3s:.3f} m³/s", delta=f"Cost: {opt_plan.total_cost:.1f}")
        gap = report.gap("greedy")
        gap_str = f"{gap * 100:.1f}%" if gap is not None else "0.0%"
        m4.metric("Greedy Optimality Gap", gap_str, delta_color="inverse")

    # Table of all strategies
    strat_rows = []
    for s_name, plan in report.plans.items():
        gap = report.gap(s_name)
        gap_val = f"{gap * 100:.1f}%" if gap is not None else "—"
        strat_rows.append({
            "Strategy": s_name.capitalize(),
            "Drains Repaired": str(list(plan.repaired_drain_ids)),
            "Total Cost": f"{plan.total_cost:.1f} / {budget:.1f}",
            "Overflow Reduction (m³/s)": round(plan.overflow_reduction_m3s, 4),
            "Residual Overflow (m³/s)": round(plan.total_overflow_m3s, 4),
            "Optimality Gap": gap_val,
        })
    st.table(pd.DataFrame(strat_rows))

    # Strategy Chart
    st.pyplot(plot_strategy_comparison(report))

    st.markdown(
        """
        **Algorithmic Insight:**
        - **Greedy Repair:** Iteratively chooses the drain yielding the largest marginal overflow reduction in the
          *current state* that fits within remaining budget. Fast ($O(m \\cdot \\text{maxflow})$), but can be suboptimal
          due to combinatorial path interactions.
        - **Exhaustive Optimization:** Evaluates all $2^n$ feasible candidate subsets. Always achieves the provably
          global optimal allocation, but exhibits exponential runtime growth.
        """
    )


# ---------------------------------------------------------------------------
# Section F: Results & Audit
# ---------------------------------------------------------------------------

def render_results_audit(net, scenarios) -> None:
    st.header("📊 Results, Bottlenecks & Data Provenance Audit")
    st.markdown("Comprehensive evaluation of model outputs, identified bottlenecks, and data provenance audit.")

    sc_choice = st.selectbox("Select Scenario to Inspect", list(scenarios.keys()), index=1)
    scenario = scenarios[sc_choice]

    fn = build_flow_network(net, scenario)
    result = analyze(net, fn, cross_check=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Max Flow Delivered", f"{result.max_flow_m3s:.3f} m³/s")
    c2.metric("Min Cut Capacity", f"{result.min_cut_capacity_m3s:.3f} m³/s")
    c3.metric("Total Overflow Shortfall", f"{result.total_overflow_m3s:.3f} m³/s")

    st.subheader("Critical Bottleneck (Minimum Cut Edges)")
    st.markdown(
        "By the Max-Flow Min-Cut Theorem, the maximum deliverable flow equals the capacity of the minimum cut. "
        "The capacity-bearing edges below form the immediate physical bottleneck in this scenario:"
    )

    cut_rows = []
    for edge in result.min_cut_edges:
        cut_rows.append({
            "Edge Kind": edge.kind.capitalize(),
            "Record Ref": edge.ref,
            "From &rarr; To": f"{edge.u} &rarr; {edge.v}",
            "Edge Capacity (m³/s)": round(edge.capacity, 3),
        })
    st.table(pd.DataFrame(cut_rows))

    st.markdown("---")
    st.subheader("Data Provenance & Assumption Ledger")
    provenance_table = [
        {"Asset / Dimension": "IMD Historical Rainfall", "Category": "REAL:IMD_GRID", "Verified Source": "IMD / OpenCity (1901–2021)", "Impact on Results": "Supplies empirical rainfall distribution"},
        {"Asset / Dimension": "Waterlogging Hotspots", "Category": "DERIVED:GEOCODED", "Verified Source": "MCGM Civic Reports", "Impact on Results": "Provides reference junction coordinates"},
        {"Asset / Dimension": "Drain Network Nodes & Edges", "Category": "SYNTHETIC", "Verified Source": "FloodCut Testbed Model", "Impact on Results": "Defines graph topology and conduit capacities"},
        {"Asset / Dimension": "Runoff Coefficients & Areas", "Category": "ASSUMED", "Verified Source": "Civil Engineering Assumptions", "Impact on Results": "Determines catchment peak runoff via Rational Method"},
        {"Asset / Dimension": "Outfall Tidal Reductions", "Category": "ASSUMED", "Verified Source": "Scenario Tide Factors [0, 1]", "Impact on Results": "Scales sea and river outfall discharge caps"},
    ]
    st.table(pd.DataFrame(provenance_table))


if __name__ == "__main__":
    main()
