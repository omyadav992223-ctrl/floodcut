# Architecture

## Technology stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Readable, strong graph tooling |
| Graph library | NetworkX | Graph structures, and an independent reference max-flow (`preflow_push`) that `maxflow.analyze()` cross-checks our own Edmonds–Karp against |
| Models | stdlib `dataclasses` | No extra dependency; immutable; self-validating |
| Data format | JSON | Human-readable, easy to inspect in a viva |
| Tests | pytest | |
| Later phases only | Streamlit + Folium (dashboard/map), optionally pandas, OSM/Open-Meteo ingestion | Not installed or used in Phase 1 |

Phase 1 runtime dependency: **networkx** only.

## Layers

```
data/*.json --io.py--> models.py (DrainageNetwork) --validation.py--> Issues
                              |
                              +--graph.py--> physical MultiDiGraph
                              +--graph.py--> flow network (S*, T*) --maxflow.py--> FloodCutResult
                                                                                       |
                                                                        cli.py <-------+--- user
```

| Module | Responsibility | Must not |
|---|---|---|
| `models.py` | Immutable records; scalar validation at construction | contain graph logic |
| `hydrology.py` | Rational-method demand Q = C·i·A/360 | know about graphs |
| `io.py` | Load/save JSON; mandatory `is_simulated` provenance flag | validate cross-record rules |
| `validation.py` | Cross-record checks; returns ERROR/WARNING issues | raise on first problem |
| `graph.py` | Build physical graph and flow network | compute flow |
| `maxflow.py` | Own Edmonds–Karp max flow, min cut, overflow (Phase 2) | mutate `DrainageNetwork`/`FlowNetwork`; it only reads them |
| `cli.py` | `info`, `validate`, `demand`, `maxflow` | hold logic |

## Core data models

`Junction`, `Drain`, `Catchment`, `Outfall`, `CriticalAsset`, `RainfallScenario`, `NetworkMetadata`, and the
`DrainageNetwork` container. Units: m³/s, hectares, mm/h, abstract cost units. Drain effective capacity =
`design_capacity_m3s × capacity_factor`. Outfall effective capacity = `base × tide_factor` if tidal.
`CriticalAsset` is in the schema but not analysed until Phase 5.

## Flow-network encoding

```
S* --[q_c]--> catchment:C --(inf)--> J_in --[cap_d]--> drain:D --(inf)--> J_out --[cap_o]--> T*
```

Each drain becomes a node with one capacity-bearing edge, so parallel drains stay distinct (a plain `DiGraph`
would merge them) and a min cut maps straight back to drain ids through the `ref` edge attribute. Edges without a
`capacity` attribute are infinite in NetworkX. Node capacities (pumps, junction limits) are not modelled yet.

## Max-flow engine (Phase 2)

`maxflow.py` implements Edmonds–Karp (BFS-selected Ford–Fulkerson) itself, from scratch,
against the `nx.DiGraph` `graph.build_flow_network` already produces — it does not call
any NetworkX flow function for the result it reports. NetworkX's own max flow
(`preflow_push` by default, a different algorithm family) is used only as an independent
cross-check inside `analyze()`. See `docs/MAXFLOW.md` for the full write-up: the formulas,
the max-flow/min-cut theorem argument, the worked numeric example on the bundled dataset,
and the three-pronged validation strategy (hand computation, NetworkX cross-check,
brute-force min-cut enumeration).

## Design decisions

1. Physical graph is a `MultiDiGraph` (parallel drains are real); the flow network is a `DiGraph` (NetworkX max-flow requires it).
2. Scalar validity is enforced in models; relational validity in `validation.py`.
3. `is_simulated` has no default: a dataset cannot be loaded without declaring its provenance.
4. Ids may not contain `:` or `*`, which are reserved for graph node names and S*/T*.
