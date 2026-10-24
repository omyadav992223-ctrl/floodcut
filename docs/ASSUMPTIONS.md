# Modelling assumptions and known limitations

Read this before trusting any output. FloodCut is a **capacity-screening model**, not a flood predictor.

| # | Assumption | Consequence |
|---|---|---|
| A1 | Steady state: one snapshot, no storage or time | Cannot represent ponding depth or duration (a time-expanded network is a later option) |
| A2 | Rational method Q = C·i·A/360 with uniform intensity | No time of concentration; coefficients are invented |
| A3 | Flow can be routed optimally (max-flow semantics) | Real water follows gravity and hydraulics, so max flow is a best-case upper bound |
| A4 | Drain capacity is constant per scenario | Real capacity varies with slope, depth, silt, and tide backwater |
| A5 | Tide only scales outfall capacity by `tide_factor` | No reverse flow or tide-dependent interior capacity |
| A6 | Edge capacities only | No pumps, junction limits, or storage |
| A7 | Overflow *location* is not unique under max flow | **Phase 2 rule (see `docs/MAXFLOW.md` §6, `maxflow.py` §5):** catchment overflow = demand − flow on that catchment's own demand edge, read off the one maximum flow Edmonds–Karp returns. The scenario *total* overflow is unique; the *breakdown* across catchments sharing a downstream bottleneck is not, and `FloodCutResult` states this every time via `note_on_overflow_uniqueness` |
| A8 | Min cuts are not unique; being in one does not imply a repair helps | **Phase 2 status:** `analyze()` reports *a* minimum cut (via final-residual-graph reachability), explicitly documented as not necessarily *the* only one. Enumerating *all* min cuts and turning cut membership into a repair recommendation is still Phase 4's job (marginal-gain greedy, not cut membership) |
| A9 | All bundled data is synthetic | Results describe the model only |
| A10 | Repair cost is in abstract units | No rupee figures are claimed |
