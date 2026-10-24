# FloodCut: Mumbai Flood Risk & Drainage Optimization

A Discrete Mathematics project: model a drainage network as a directed capacitated graph, find bottlenecks with
Max-Flow Min-Cut, analyse combinations of drain failures, and choose repairs under a budget.

**Status: Phase 7 (Testing and Final Audit) complete.** The complete 7-phase roadmap is fully implemented,
verified by 275 automated offline unit and integration tests, and mathematically audited for flow conservation,
capacity bounds, and min-cut duality.

> **Academic Prototype Notice:** The IMD rainfall data in `data/real/` is a real observed dataset (1901–2021).
> The drainage network geometry and capacities in `data/simulated/` are **SYNTHETIC** testbed models.
> Results from this project describe a prototype capacity-screening model, not Mumbai's real flood risk.
> See `docs/PHASE5_DATA_INTEGRATION.md#known-limitations`.

---

## Discrete Mathematics Syllabus Alignment

FloodCut directly maps core syllabus concepts to working, tested software:

- **Graph Theory:** Directed capacitated networks $G = (V, E, c)$, vertices, confluences, and flow conservation.
- **Max-Flow Min-Cut Theorem:** Edmonds–Karp residual graph augmentations and bottleneck identification.
- **Combinatorics:** Combinatorial failure enumeration across $\binom{m}{k}$ subsets.
- **Set Theory:** Cut partitions $(S, T)$ and feasible repair subset lattices.
- **Combinatorial Optimization:** Budget-constrained 0-1 knapsack-style repair optimization.
- **Algorithm Analysis:** Marginal-gain Greedy heuristic ($O(m \cdot T_{\text{flow}})$) vs global Exhaustive search ($O(2^n \cdot T_{\text{flow}})$) with optimality gap quantification.

👉 **Full Mathematical Write-up:** See [`docs/DISCRETE_MATH_ALIGNMENT.md`](docs/DISCRETE_MATH_ALIGNMENT.md).

---

## Setup & Reproduction Instructions (Python 3.10+)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Validation & Tests

```bash
# 1. Validate the drainage network data
floodcut validate

# 2. Run the complete automated test suite (275 tests, ~10s runtime offline)
pytest -v
```

### 3. Launch the Interactive Dashboard

Launch the Streamlit web dashboard locally:

```bash
floodcut app                       # launches on http://localhost:8501
floodcut app --port 8502           # custom port
floodcut dashboard                 # alias for app
```

Or run Streamlit directly:

```bash
streamlit run src/floodcut/app.py
```

---

## Dashboard Walkthrough

1. **Project Overview:** Problem description, graph metrics, provenance badges, and prototype disclaimers.
2. **Interactive Map:** Leaflet/Folium pan-zoom map displaying real IMD grid bounds (`REAL:IMD_GRID`), civic waterlogging hotspots (`DERIVED:GEOCODED`), and synthetic drain conduits (`SYNTHETIC`). Includes on-map legend and offline topological graph fallback.
3. **Rainfall Explorer:** 121-year historical IMD monthly data (1901–2021), annual and monsoon trends, and monthly breakdown charts.
4. **Failure Simulation:** Interactive selection of blocked drains, real-time max flow and overflow shortfall re-calculation, catchment breakdown, and automated worst-case $k$-drain failure enumeration ($k \in \{1, 2\}$).
5. **Repair Optimization:** Interactive budget slider, degraded drain candidates inspection, real-time execution of Greedy heuristic vs global Exhaustive search, baselines comparison, and optimality gap metrics.
6. **Results & Audit:** Critical bottleneck (minimum cut edges) identification and data provenance ledger.

---

## Command-Line Interface (CLI)

```bash
floodcut validate                          # validate network JSON data
floodcut info                              # counts, DAG check, parallel drains, degraded drains
floodcut demand --scenario design          # rainfall demand per catchment (50 mm/h)
floodcut demand --scenario extreme         # 100 mm/h with high tide
floodcut maxflow --scenario design         # max flow, min cut, overflow (Phase 2 engine)
floodcut maxflow --scenario extreme --no-cross-check   # skip NetworkX cross-check
floodcut failure -k 2 --scenario design    # 2-drain failure simulation
floodcut repair --budget 50 --scenario design   # repair optimisation
floodcut app                               # start web dashboard
python -m floodcut info                    # same, without console script
```

---

## Data Provenance Architecture

| Directory / File | Provenance Label | Description |
|---|---|---|
| `data/real/imd_mumbai_monthly_rainfall_1901_2021.csv` | `REAL:IMD_GRID` | 121 years of monthly observations from IMD via OpenCity India |
| `src/floodcut/geo_loader.py` (IMD Grid Point) | `REAL:IMD_GRID` | IMD 0.25° grid cell centre ($19.00^\circ\text{ N}, 72.75^\circ\text{ E}$) |
| `src/floodcut/geo_loader.py` (Hotspots) | `DERIVED:GEOCODED` | 7 waterlogging junctions geocoded from MCGM civic reports |
| `data/simulated/mumbai_sim_network.json` | `SYNTHETIC` | 14-junction, 14-drain illustrative benchmark network |
| `data/simulated/rainfall_scenarios.json` | `SYNTHETIC` | Benchmark storm intensities & tide factors |

---

## Repository Layout

```
src/floodcut/
  app.py                         # Streamlit interactive web dashboard (Phase 6)
  ui/                            # UI helper modules (Phase 6)
    map_view.py                  # Folium map & offline graph
    charts.py                    # Matplotlib visualizers
    state.py                     # Cached data loaders
  models.py                      # Immutable domain dataclasses with scalar validation
  hydrology.py                   # Rational method demand calculation Q = C*i*A/360
  io.py                          # JSON network & scenario I/O
  validation.py                  # Relational cross-record validation
  graph.py                       # Physical MultiDiGraph & FlowNetwork (S*, T*)
  maxflow.py                     # Edmonds–Karp max-flow & min-cut engine
  failure.py                     # Combinatorial k-failure enumeration
  repair.py                      # Greedy, exhaustive, & baseline repair optimization
  cli.py                         # Command-line interface
  provenance.py                  # Data provenance labels & metadata
  rainfall_loader.py             # IMD monthly rainfall CSV loading & unit conversion
  geo_loader.py                  # Geospatial points loading & validation
data/
  simulated/                     # Synthetic testbed datasets
  real/                          # Real IMD observational datasets
tests/                           # 275 automated offline tests
docs/
  ARCHITECTURE.md                # System layers & design decisions
  ROADMAP.md                     # Completed 7-phase roadmap
  ASSUMPTIONS.md                 # 10 modeling assumptions & limitations
  MAXFLOW.md                     # Max-flow formulas, proofs & worked examples
  PHASE5_DATA_INTEGRATION.md     # Data sources, units, & provenance
  PHASE6_FRONTEND.md             # Dashboard architecture & setup
  DISCRETE_MATH_ALIGNMENT.md     # Discrete Mathematics syllabus mapping
```

---

## Troubleshooting & Known Limitations

1. **CartoDB Basemap Warning:** If Folium issues a warning regarding CartoDB tiles, the map has been preconfigured to use OpenStreetMap (`OpenStreetMap`) tiles without requiring an API key.
2. **Windows Terminal CP1252:** All CLI outputs use ASCII arrows (`->`) to ensure clean execution across Windows Command Prompt and PowerShell without encoding errors.
3. **Monthly Rainfall Disaggregation:** Monthly rainfall accumulated millimetres (mm) converted to mean hourly intensity (mm/h) produces a time-averaged rate, NOT a peak design-storm intensity. Use scenario intensities (e.g., 50 mm/h) for flood capacity screening.
4. **Offline Tests:** The entire test suite (`pytest`) is designed to run offline without internet connectivity.
