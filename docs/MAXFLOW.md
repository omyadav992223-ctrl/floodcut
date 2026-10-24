# Phase 2: the max-flow engine

This is a companion to the full explanation already embedded as the module docstring
of `src/floodcut/maxflow.py` (read that first; this page adds a worked numeric example
and the test-strategy summary a reviewer would want). All formulas below are restated,
not just referenced, so this page is self-contained.

## 1. What problem is being solved

Phase 1 encodes one rainfall scenario as a flow network (`graph.build_flow_network`):

```
S* --[q_c]--> catchment:C --(inf)--> J_in --[cap_d]--> drain:D --(inf)--> J_out --[cap_o]--> T*
```

Phase 2 answers three questions about that network:

1. **Maximum flow** — the largest total discharge (m^3/s) that can reach T* at once,
   obeying every edge's capacity. This is the network's best-case deliverable flow.
2. **Minimum cut** — the smallest-capacity set of edges whose removal would disconnect
   S* from T*. By the max-flow min-cut theorem this always has the same total capacity
   as the maximum flow, and it identifies *which* drains/catchment demand/outfalls are
   the system's bottleneck.
3. **Overflow** — demand that could not be delivered: `total_demand - max_flow`, and,
   per catchment, `demand(C) - flow(S*, catchment:C)`.

## 2. The algorithm: Edmonds–Karp

Edmonds–Karp is Ford–Fulkerson's augmenting-path method, specialised to always pick the
*shortest* (fewest-edge) augmenting path, found by BFS on the residual graph. Pseudocode:

```
flow(e) = 0 for every edge e
while BFS(residual graph) finds a path P from source to sink:
    bottleneck = min residual capacity of any edge on P
    for each edge (u, v) on P:
        flow(u, v) += bottleneck
        flow(v, u) -= bottleneck      # the paired reverse/residual edge
return sum of flow(S*, v) over all v   # = the max flow value
```

Why BFS specifically: an arbitrary-path Ford–Fulkerson can need an unbounded number of
augmentations on adversarial capacities. Always taking the shortest path bounds the
number of augmentations by O(V·E) (classical result: each edge can be "critical" —
i.e. the bottleneck — at most O(V) times across the whole run before it leaves the
residual graph for good, and there are O(E) edges), each BFS costs O(E), giving an
O(V·E^2) worst case overall. `FloodCutResult.iterations` exposes the actual augmentation
count for any run, so this bound is directly checkable rather than just asserted.

Our implementation (`maxflow.edmonds_karp`) builds its own adjacency-list residual
graph from the `networkx.DiGraph` Phase 1 already builds, rather than depending on
NetworkX's own flow algorithms for anything load-bearing — NetworkX is used only
afterwards, as an *independent* check (section 4).

## 3. The max-flow / min-cut theorem

For any flow f and any s-t cut (S, T): `|f| <= capacity(S, T)` (weak duality — every
unit of flow has to cross the cut somewhere, and the cut's capacity bounds the total
that can cross). Ford & Fulkerson (1956) showed equality is achieved at the optimum:
when Edmonds–Karp's BFS can no longer reach t, take S = {nodes still reachable from s
in the final residual graph}. Every S→T edge in the original graph must be saturated
(otherwise its residual capacity would be positive, and its head would be reachable,
contradicting its being in T), and every T→S edge must carry zero flow (otherwise its
reverse residual edge would pull its tail into S). So the cut's capacity equals exactly
the flow crossing it, which equals `|f|` by conservation — and an equal flow and cut
capacity, combined with weak duality, means both are simultaneously optimal.

`maxflow._min_cut_from_residual` implements this directly: one more BFS from the
source over the *finished* residual graph.

## 4. Independent validation strategy

Three independent lines of evidence, matching the project roadmap's Phase 2 exit
criteria, all in `tests/test_maxflow.py`:

| Check | What it validates | Where |
|---|---|---|
| Hand-computed examples | Our arithmetic on small, manually-traced networks (series bottleneck, parallel-sum, mid-diamond bottleneck, and a full worked trace of the bundled 14-junction dataset) | `test_hand_computed_*`, `test_sim_network_design_scenario_hand_verified` |
| Cross-check vs NetworkX | Our *algorithm* (Edmonds–Karp) against a different algorithm family (`networkx.maximum_flow_value`, default `preflow_push`) on the same graphs, including every bundled scenario and 35 randomized graphs (some with cycles, disconnection, zero capacities, anti-parallel edges) | `test_all_bundled_scenarios_cross_check_against_networkx`, `test_random_*` |
| Brute-force min cut | The max-flow value against the *definition* of minimum cut (literal enumeration of all 2^(n-2) S/T partitions), independent of any flow algorithm at all | `brute_force_min_cut`, `test_hand_computed_diamond_bottleneck_in_the_middle`, `test_random_small_graphs_match_networkx_and_brute_force` |

Every `analyze()` call also self-checks: it recomputes the reported min cut's capacity
from the edges it found and raises `FloodCutError` if it does not equal the max flow to
within floating-point tolerance — this is the max-flow/min-cut theorem treated as an
*invariant to assert*, not just a property we hope holds.

## 5. Worked example: the bundled dataset, "design" scenario

Reproduced in full, with the arithmetic, in the docstring of
`test_sim_network_design_scenario_hand_verified`. Summary: the dataset's 14 junctions
form two drainage basins joined by one cross-link drain (`D07`, `J06 -> J10`). At 50 mm/h:

- Total demand (six catchments, rational method): **7.472 m^3/s**
- OUT1 basin bottleneck: its own trunk (`D08a` + `D08b`, 3.0 + 0.0 since `D08b` is fully
  blocked) caps it at exactly **3.000 m^3/s**, even though upstream capacity is more
  generous — the surplus (1.2 m^3/s) is instead routed sideways via `D07` to help OUT2.
- OUT2 basin: its bottleneck is `D11` (capacity 1.6) once it receives that rerouted 1.2
  m^3/s on top of catchment C5's own demand; it ends up delivering **2.378 m^3/s**.
- **Max flow = 3.000 + 2.378 = 5.378 m^3/s; overflow = 7.472 - 5.378 = 2.094 m^3/s**
  (28% of demand), matching both the Edmonds–Karp result and the NetworkX cross-check
  to 6 decimal places.

Run it yourself:

```bash
floodcut maxflow --scenario design
```

## 6. What Phase 2 deliberately does **not** claim

Per `docs/ASSUMPTIONS.md`:

- **A7 (overflow location)**: catchment-level overflow and outfall-level delivered flow
  are read off *one* maximum flow. The scenario-wide total is unique; the breakdown
  across catchments sharing a downstream bottleneck is not, and `FloodCutResult` says so
  via `note_on_overflow_uniqueness` every time.
- **A8 (min-cut uniqueness)**: `analyze()` reports *a* minimum cut, never *the* minimum
  cut. Several different edge sets can tie for the minimum capacity; being on this run's
  reported cut does not by itself imply a repair there is the only — or even the best —
  option (that marginal-gain comparison is Phase 4's job).
- Flow is still a steady-state, optimally-routed upper bound (A1, A3): real stormwater
  does not get centrally dispatched the way max-flow routes it.
