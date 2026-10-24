# Phase 6: Frontend and Visualization

## Overview

Phase 6 introduces an interactive, modular web dashboard for **FloodCut**, connecting the discrete mathematical algorithms (Edmonds–Karp maximum flow, min-cut bottleneck detection, combinatorial failure simulations, and budget-constrained repair optimization) and Phase 5 real-data layers to an intuitive user interface.

The dashboard is built using **Streamlit** and **Folium**, supported by **Matplotlib** for analytical charting and offline topological graph rendering.

> **Academic Prototype Notice:** FloodCut is an academic capacity-screening prototype. It models drainage networks as capacitated directed graphs to compute maximum deliverable flow and identify critical bottlenecks. The drainage network topology and capacities are **SYNTHETIC** testbed models and do NOT represent verified municipal underground infrastructure or real-world flood inundation depths.

---

## Architecture & Technology Stack

| Component | Technology | Responsibility |
|---|---|---|
| **Web Framework** | [Streamlit](https://streamlit.io/) | Interactive dashboard layout, sliders, multi-selects, tabs, and reactive metric widgets. |
| **Geospatial Mapping** | [Folium](https://python-visualization.github.io/folium/) / [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium) | Interactive pan/zoom map rendering real IMD grid bounds, waterlogging hotspots, and synthetic conduits with on-map legends and detailed popups. |
| **Offline Graph Viz** | [NetworkX](https://networkx.org/) + [Matplotlib](https://matplotlib.org/) | Fallback topological graph visualization when online map tiles cannot be fetched. |
| **Analytical Charting** | [Matplotlib](https://matplotlib.org/) | Rainfall trend lines, monthly distributions, flow comparison bars, and repair strategy Pareto charts. |
| **Core Calculation Engine** | FloodCut Backend (`models`, `maxflow`, `failure`, `repair`) | Pure Python algorithms executed directly without frontend simplifications or dummy data. |

---

## Directory Structure

```
src/floodcut/
├── app.py              # Main Streamlit application entry point
├── ui/
│   ├── __init__.py
│   ├── state.py        # Cached data loaders for network, scenarios, rainfall, and geo layers
│   ├── map_view.py     # Folium interactive map & offline network graph
│   └── charts.py       # Matplotlib visualization generators
├── cli.py              # Extended with `floodcut app` and `floodcut dashboard` commands
docs/
└── PHASE6_FRONTEND.md  # This documentation file
tests/
└── test_phase6_frontend.py # Automated tests for UI loaders, maps, charts, and CLI app
```

---

## Launch Instructions

### 1. Installation

Ensure runtime dependencies are installed:

```bash
pip install -r requirements.txt
# or with optional UI extra:
pip install -e ".[ui,dev]"
```

Dependencies added in Phase 6:
- `streamlit>=1.30.0`
- `folium>=0.15.0`
- `streamlit-folium>=0.17.0`
- `matplotlib>=3.7.0`
- `pandas>=1.5.0`

### 2. Launching the Dashboard

You can start the dashboard in either of two ways:

#### Option A: Via the FloodCut CLI
```bash
floodcut app
# or with custom port:
floodcut app --port 8501
# alias:
floodcut dashboard
```

#### Option B: Via Streamlit Directly
```bash
# From the repository root:
streamlit run src/floodcut/app.py
```

The application will be accessible in your web browser at:
`http://localhost:8501`

---

## Interface Sections

### A. Project Overview
- Displays project title, mission statement, and discrete mathematics formulation (flow networks, Rational Method, max-flow min-cut theorem, subadditivity/superadditivity, and knapsack optimization).
- Prominent system metrics: Junction count, Drain count, Catchment count, and Outfall count.
- Clear data provenance badges (`REAL:IMD_GRID`, `DERIVED:GEOCODED`, `SYNTHETIC`).
- Notice reinforcing that model overflow represents unrouted discharge in an academic screening model, not measured inundation depths.

### B. Interactive Map
- Visualizes three distinct spatial layers:
  1. **IMD Rainfall Grid Station (`REAL:IMD_GRID`):** Green marker and bounding box for the official IMD $0.25^\circ \times 0.25^\circ$ grid cell ($19.00^\circ\text{ N}, 72.75^\circ\text{ E}$).
  2. **Waterlogging Hotspots (`DERIVED:GEOCODED`):** Amber warning markers for 7 chronic waterlogging junctions compiled from MCGM reports (Hindmata, Milan Subway, King Circle, Andheri Subway, Sion Circle, Kurla, Matunga Road).
  3. **Drainage Network (`SYNTHETIC`):** Blue conduits for healthy drains, dashed red lines for degraded/blocked drains, blue circles for network junctions, and green nodes for tidal sea/river outfalls.
- Layer toggles to isolate specific features.
- Fixed on-map HTML legend distinguishing feature types.
- Popups displaying effective capacity, healthy factor, repair cost, and provenance.
- **Offline Mode:** Includes a toggle to switch to a standalone Matplotlib-rendered network graph if internet connectivity is unavailable.

### C. Rainfall Explorer
- Explores 121 years of observational monthly rainfall data (1901–2021) from IMD / OpenCity.
- Year range slider and year-specific dropdown selector.
- Annual total and monsoon total (Jun–Sep) time-series trend chart with historical average reference line.
- Monthly distribution bar chart highlighting the monsoon concentration (green) versus non-monsoon months (blue).
- Filterable tabular view displaying all monthly records with provenance tags.

### D. Failure Simulation
- Directly executes Phase 3 combinatorial failure calculations.
- Select baseline scenario (`light`, `design`, `heavy`, `extreme`).
- Interactive multiselect to completely block any combination of healthy or degraded drains.
- Instant re-calculation of maximum deliverable flow, simulated overflow, and marginal added shortfall ($\Delta \text{Overflow}$).
- Catchment demand vs delivered flow vs overflow shortfall breakdown table.
- Automated worst-case $k$-drain failure enumeration ($k=1, 2$) displaying evaluated subset counts ($C(m, k)$) and identified worst-case combinations.

### E. Repair Optimization
- Directly executes Phase 4 repair prioritization algorithms under budget $B$.
- Displays degraded drains (`capacity_factor < 1.0`), their repair cost, and current condition.
- Interactive slider to set the available repair budget.
- Side-by-side strategy comparison table evaluating:
  * **Greedy Repair:** Marginal-gain greedy heuristic re-evaluated at each step.
  * **Exhaustive Optimization:** Exact combinatorial search evaluating all feasible candidate subsets.
  * **Severity Baseline:** Repairing highest-overflow contributors first.
  * **Cheapest Baseline:** Repairing lowest-cost candidates first.
  * **Ratio Baseline:** Fractional knapsack ranking by gain-to-cost ratio.
  * **Random Baseline:** Best plan among random feasible samples.
- Calculates and displays the **Greedy Optimality Gap** percentage relative to the global optimum.
- Matplotlib Pareto chart comparing overflow reduction vs budget expenditure across all strategies.

### F. Results & Provenance Audit
- Bottleneck (Minimum Cut) edge table showing which conduits or outfalls constrain system capacity in the selected scenario.
- Comprehensive provenance audit ledger detailing every data asset, its classification, source, and impact on simulation results.

---

## Backend Connection & Integrity

1. **No Frontend Estimations:** All values for maximum flow, overflow, min cuts, failure subsets, and repair selections are computed by calling the Phase 1–5 functions (`analyze()`, `_apply_failures()`, `enumerate_failures()`, `compare_strategies()`, `greedy_repair()`, `exhaustive_repair()`).
2. **Deterministic Offline Execution:** All state loaders fallback safely to the bundled synthetic fixtures if external files are missing.
3. **Data Provenance Preservation:** Every screen, chart, and table explicitly prints provenance tags (`REAL`, `DERIVED`, `ASSUMED`, `SYNTHETIC`) so users never mistake model assumptions for real-world measurements.

---

## Verification & Testing

Run the automated test suite:

```bash
# Test Phase 6 components specifically:
python -m pytest -v tests/test_phase6_frontend.py

# Run the complete test suite (259 tests):
python -m pytest -q
```

All 259 tests pass cleanly in ~8.9 seconds without requiring live internet access.
