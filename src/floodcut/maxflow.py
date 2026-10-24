"""Phase 2: the max-flow / min-cut mathematical engine.

This module answers, for one rainfall scenario, the question Phase 1 only set up:
*given the demand the catchments put into the network and the capacities of the
drains and outfalls, how much water can actually get out, where is the network's
bottleneck, and how much overflows?*

-------------------------------------------------------------------------------
1. The max-flow problem, formally
-------------------------------------------------------------------------------
Let G = (V, E) be a directed graph with a capacity c(e) >= 0 on every edge, a
source s in V and a sink t in V (s != t). A **flow** is a function f: E -> R
such that:

    (capacity)    0 <= f(u, v) <= c(u, v)                      for every edge (u, v)
    (conservation) sum_{(u,v) in E} f(u,v) = sum_{(v,w) in E} f(v,w)   for every v != s, t

i.e. every node except the source and sink passes on exactly what it receives.
The **value** of a flow is the net amount leaving the source:

    |f| = sum_{(s,v) in E} f(s,v)  -  sum_{(u,s) in E} f(u,s)

In our flow network (`graph.build_flow_network`) the super-source S* has no
incoming edges, so |f| is simply the total flow leaving S*. The **max-flow
problem** is to find an f that maximises |f|. In FloodCut this value is the
total deliverable discharge, in m^3/s, for the scenario: the most water the
network could carry out to the outfalls if every drop were routed optimally
(see docs/ASSUMPTIONS.md, A3 — real water does not route optimally, so this is
a best-case upper bound, not a prediction).

-------------------------------------------------------------------------------
2. Residual graphs and augmenting paths (Ford-Fulkerson method)
-------------------------------------------------------------------------------
Given a flow f, the **residual graph** G_f has, for every edge (u, v) in E:

    a forward residual edge  u -> v  with residual capacity  c(u,v) - f(u,v)
    a backward residual edge v -> u  with residual capacity  f(u,v)

The backward edge lets the algorithm "undo" flow it sent earlier if a better
routing is later discovered — this is what makes the method correct even when
the first path chosen is not part of any optimal solution.

An **augmenting path** is a path from s to t in G_f using only edges with
strictly positive residual capacity. Its **bottleneck** is the minimum
residual capacity along it. The Ford-Fulkerson *method* is: while an
augmenting path exists, push its bottleneck amount of flow along it (adding to
forward edges, subtracting from the paired backward edges) and repeat. It
terminates with a maximum flow whenever every capacity is finite (true here:
demand, drain and outfall edges always carry a finite non-negative capacity;
only the catchment-link and drain-link edges are capacity-less / infinite, and
those never bound the bottleneck of any real path because a finite edge always
sits in series with them).

-------------------------------------------------------------------------------
3. Edmonds-Karp: Ford-Fulkerson with a fixed path rule
-------------------------------------------------------------------------------
Ford-Fulkerson does not say *which* augmenting path to use, and a careless
choice can need a very large number of augmentations. **Edmonds-Karp** is the
specific, classic rule: always augment along a **shortest** augmenting path
(fewest edges), found by breadth-first search (BFS) on the residual graph.
This guarantees the method finishes in at most O(V * E) augmentations (a
classical result: each edge can be the bottleneck of a shortest path only
O(V) times before it leaves the residual graph for good), each BFS costing
O(E), for a total worst case of O(V * E^2). That bound is what makes the
algorithm "demonstrable": it is simple, deterministic, and its termination
does not depend on the capacities being integers or being chosen luckily.

`_edmonds_karp` below is the implementation: a BFS to find the next shortest
augmenting path, then an augmentation along it, repeated until BFS finds no
path from source to sink at all.

-------------------------------------------------------------------------------
4. The max-flow / min-cut theorem, and how we read off a minimum cut
-------------------------------------------------------------------------------
An s-t **cut** is a partition of V into S (containing s) and T = V - S
(containing t). Its **capacity** is the sum of c(u, v) over edges with u in S,
v in T (edges the "wrong way", T to S, are not counted). The **max-flow
min-cut theorem** (Ford & Fulkerson, 1956) states:

    max over flows f of |f|   =   min over cuts (S,T) of capacity(S,T)

*Weak duality* (|f| <= capacity(S,T) for every flow and every cut) is easy:
every unit of flow leaving S for T must cross the cut, and the cut's capacity
bounds that total. The *equality* at the optimum is what Edmonds-Karp proves
constructively: when BFS can no longer reach t from s in the residual graph,
let S = {nodes reachable from s in the final residual graph} and T = V - S.
Every edge from S to T must then be **saturated** (f(u,v) = c(u,v)) — otherwise
its residual capacity would be positive and v would be reachable, putting it
in S, a contradiction. Every edge from T to S must carry **zero flow** —
otherwise its backward residual edge (capacity f(u,v) > 0) would make its
S-side endpoint reachable from the T-side node that is reachable from... more
directly: if f(u,v) > 0 for u in T, v in S, the backward edge v -> u has
residual capacity f(u,v) > 0, which (combined with S being reachable) would
put u in S too — contradiction. So capacity(S,T) = sum of f(u,v) over the
forward cut edges = the flow actually crossing the cut = |f|. Hence this
particular (S,T) is simultaneously a flow value and a cut capacity that are
equal, which by weak duality means both are optimal.

`_min_cut_from_residual` implements exactly this: BFS once more from the
source in the final residual graph, take S = reached nodes, and report the
original (finite-capacity) edges leaving S. Per docs/ASSUMPTIONS.md A8, a
minimum cut is **not** generally unique (several different edge sets can all
have the theorem's minimum capacity); we therefore only claim to report *a*
minimum cut (the one this specific run of BFS finds), never *the* minimum cut.
Tests validate the *value* exhaustively, by brute-force cut enumeration on
small graphs (`brute_force_min_cut`), and leave uniqueness unclaimed.

-------------------------------------------------------------------------------
5. Overflow: the one place we *do* commit to a definition (A7)
-------------------------------------------------------------------------------
docs/ASSUMPTIONS.md (A7) warns that overflow *location inside the network* is
not unique under max flow: when a bottleneck is upstream, max flow is free to
"choose" which of several catchments' water gets through and which backs up,
and a different maximum flow could choose differently while carrying the same
total value. FloodCut resolves this ambiguity for **catchments** (not for
interior junctions) with one explicit, well-defined rule:

    overflow(catchment C) = demand(C) - f(S*, catchment:C)

i.e. overflow is read directly off the flow actually placed on *that
catchment's own demand edge* in the max flow Edmonds-Karp returns. This number
is well-defined for any flow f (it only needs f, not a choice among several
maximum flows), but *is* a property of the specific maximum flow computed, since
alternate maximum flows of equal total value can distribute shortfall across
catchments differently whenever the bottleneck is downstream of where two
catchments' paths merge. We therefore document this explicitly here and in the
returned `FloodCutResult.note_on_overflow_uniqueness` rather than claim more
than is true. The same honest caveat applies to `outfall_delivered`. The one
quantity that *is* unique regardless of which maximum flow is chosen is the
scenario total: total_overflow_m3s = total_demand_m3s - max_flow_value, because
every maximum flow has the same value (that is what "maximum" means).

-------------------------------------------------------------------------------
6. Edge cases this module is required to handle
-------------------------------------------------------------------------------
- **Disconnected graphs**: if the sink is unreachable from the source at all
  (no path exists, ignoring capacity), BFS finds no augmenting path on its very
  first call, the loop never executes, and the max flow is correctly 0 — no
  special-casing needed. The same applies to a single catchment whose junction
  cannot reach any outfall (validation.py's W_CATCHMENT_NO_PATH warning): its
  demand edge simply never appears on any augmenting path, so its flow stays 0
  and its overflow equals its full demand.
- **Zero capacities**: an edge with capacity 0 (e.g. a fully blocked drain,
  `capacity_factor == 0`) has residual capacity 0 from the start and is simply
  never traversed by BFS. No special-casing needed; `tests/test_maxflow.py`
  checks this on the bundled dataset's own blocked drain (D08b).
- **Invalid inputs**: negative, NaN or non-numeric capacities, a missing
  source/sink node, or source == sink are all rejected up front with a
  `FloodCutError` (see `_build_adjacency` and `edmonds_karp`) rather than
  producing a silently wrong answer.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from .graph import FlowNetwork, SINK, SOURCE, catchment_node, drain_node
from .models import DrainageNetwork

#: BFS augmentations are bounded by O(V*E); this is a generous safety valve
#: against an implementation bug turning into an infinite loop, not a limit
#: expected to be hit by any legitimate (finite, non-adversarial) network.
_MAX_ITERATIONS = 200_000

#: Flows / residual capacities are compared with this tolerance because they
#: are floating point sums of many augmentations.
_EPS = 1e-9


class FloodCutError(ValueError):
    """Raised when the max-flow engine is given a structurally invalid graph."""


# ----------------------------------------------------------------------------
# Internal residual-graph representation
# ----------------------------------------------------------------------------

@dataclass
class _Edge:
    """One directed residual-graph edge, with its paired reverse edge.

    Every original edge (u, v, capacity=c) becomes exactly one `_Edge` object
    (`is_original=True`) stored in `adj[u]`, plus a zero-capacity companion
    (`is_original=False`) stored in `adj[v]`. The two point at each other via
    `rev` (an index into the other's adjacency list), which is the standard
    O(1)-reverse-lookup trick and is what correctly keeps anti-parallel or
    multi edges independent instead of merging their capacities.
    """
    to: str
    capacity: float
    flow: float = 0.0
    rev: int = -1
    is_original: bool = True
    tail: str = ""
    kind: str = ""
    ref: str = ""

    @property
    def residual(self) -> float:
        return self.capacity - self.flow


def _build_adjacency(graph: nx.DiGraph) -> dict[str, list[_Edge]]:
    """Turn an nx.DiGraph (capacity attribute optional = infinite) into an
    adjacency-list residual graph, validating every capacity on the way."""
    adj: dict[str, list[_Edge]] = defaultdict(list)
    for node in graph.nodes:
        adj[node]  # ensure every node has an (initially empty) entry
    for u, v, data in graph.edges(data=True):
        cap = data.get("capacity", math.inf)
        if isinstance(cap, bool) or not isinstance(cap, (int, float)):
            raise FloodCutError(f"edge {u}->{v}: capacity must be a number, got {cap!r}")
        cap = float(cap)
        if math.isnan(cap) or cap < 0:
            raise FloodCutError(f"edge {u}->{v}: capacity must be >= 0, got {cap!r}")
        fwd = _Edge(to=v, capacity=cap, is_original=True, tail=u,
                    kind=data.get("kind", ""), ref=data.get("ref", ""))
        bwd = _Edge(to=u, capacity=0.0, is_original=False, tail=v)
        adj[u].append(fwd)
        adj[v].append(bwd)
        fwd.rev = len(adj[v]) - 1
        bwd.rev = len(adj[u]) - 1
    return adj


def _bfs_shortest_augmenting_path(adj: dict[str, list[_Edge]], source: str, sink: str
                                   ) -> list[tuple[str, int]] | None:
    """BFS on residual capacities. Returns the path as a list of (node, edge_index
    into adj[node]) steps from source to sink, or None if sink is unreachable.
    BFS (as opposed to DFS or an arbitrary-path Ford-Fulkerson) is exactly what
    makes this Edmonds-Karp rather than plain Ford-Fulkerson (see module docstring,
    section 3): it always returns a path with the fewest edges."""
    came_from: dict[str, tuple[str, int]] = {}
    visited = {source}
    queue: deque[str] = deque([source])
    while queue:
        u = queue.popleft()
        if u == sink:
            break
        for i, e in enumerate(adj[u]):
            if e.residual > _EPS and e.to not in visited:
                visited.add(e.to)
                came_from[e.to] = (u, i)
                queue.append(e.to)
    if sink not in visited:
        return None
    path: list[tuple[str, int]] = []
    node = sink
    while node != source:
        prev, idx = came_from[node]
        path.append((prev, idx))
        node = prev
    path.reverse()
    return path


def _augment(adj: dict[str, list[_Edge]], path: list[tuple[str, int]]) -> float:
    """Push the path's bottleneck (minimum residual capacity along it) of flow,
    updating every forward edge and its paired reverse edge. Returns the amount pushed."""
    bottleneck = min(adj[node][idx].residual for node, idx in path)
    if not math.isfinite(bottleneck):
        # Can only happen if source and sink are connected solely by infinite
        # (link) edges with no finite-capacity edge anywhere on the path, which
        # cannot occur in a FloodCut flow network (every S*->T* path passes
        # through at least one demand edge and one outfall edge, both finite) —
        # guarded here so a misuse of this module on a different graph fails
        # loudly rather than "succeeding" with an infinite flow value.
        raise FloodCutError("augmenting path has no finite-capacity edge; "
                             "max flow is unbounded for this graph")
    for node, idx in path:
        e = adj[node][idx]
        e.flow += bottleneck
        adj[e.to][e.rev].flow -= bottleneck
    return bottleneck


def edmonds_karp(graph: nx.DiGraph, source: str, sink: str
                  ) -> tuple[float, dict[str, list[_Edge]], int]:
    """Run Edmonds-Karp max-flow on `graph` from `source` to `sink`.

    Returns (max_flow_value, final_residual_adjacency, iteration_count).
    `iteration_count` is the number of augmenting paths used, i.e. the number
    of BFS calls that found one — reported so the O(V*E) bound is checkable.

    Raises FloodCutError if source/sink are missing, equal, or any edge has an
    invalid (negative, NaN or non-numeric) capacity.
    """
    if source not in graph:
        raise FloodCutError(f"source {source!r} is not a node of the graph")
    if sink not in graph:
        raise FloodCutError(f"sink {sink!r} is not a node of the graph")
    if source == sink:
        raise FloodCutError("source and sink must be different nodes")

    adj = _build_adjacency(graph)
    total_flow = 0.0
    iterations = 0
    while True:
        path = _bfs_shortest_augmenting_path(adj, source, sink)
        if path is None:
            break  # no augmenting path left: by the max-flow min-cut theorem, this is optimal
        total_flow += _augment(adj, path)
        iterations += 1
        if iterations > _MAX_ITERATIONS:
            raise FloodCutError(
                f"exceeded {_MAX_ITERATIONS} augmentations; this should be unreachable "
                "for a finite graph and suggests a bug, not a slow-but-correct run")
    return total_flow, adj, iterations


def _min_cut_from_residual(adj: dict[str, list[_Edge]], source: str
                            ) -> tuple[set[str], list[_Edge]]:
    """Given the *final* residual graph of a completed Edmonds-Karp run, return
    (S, cut_edges): S = nodes reachable from source in the residual graph, and
    cut_edges = original edges leaving S (see module docstring, section 4)."""
    reachable = {source}
    queue: deque[str] = deque([source])
    while queue:
        u = queue.popleft()
        for e in adj[u]:
            if e.residual > _EPS and e.to not in reachable:
                reachable.add(e.to)
                queue.append(e.to)
    cut_edges = [e for u in reachable for e in adj[u]
                 if e.is_original and e.to not in reachable]
    return reachable, cut_edges


# ----------------------------------------------------------------------------
# Brute-force min cut: an independent check, by definition rather than algorithm
# ----------------------------------------------------------------------------

def brute_force_min_cut(graph: nx.DiGraph, source: str, sink: str) -> float:
    """Minimum s-t cut capacity found by literally enumerating every subset S
    with source in S, sink not in S, and taking the minimum capacity(S, V-S).

    This is exponential (2^(n-2) subsets) and exists purely so automated tests
    can check the max-flow *value* against the min-cut *definition* directly,
    independent of any max-flow algorithm (ours or NetworkX's) -- i.e. it
    validates the max-flow/min-cut theorem itself on small graphs, not just
    one implementation of it. Intended for small test/demo graphs only.
    """
    nodes = [n for n in graph.nodes if n != source and n != sink]
    if len(nodes) > 20:
        raise FloodCutError(
            f"brute_force_min_cut is exponential and refuses graphs with > 20 "
            f"non-terminal nodes (got {len(nodes)}); use edmonds_karp instead")
    edges = [(u, v, float(d.get("capacity", math.inf))) for u, v, d in graph.edges(data=True)]
    best = math.inf
    # S always contains source and excludes sink; every other node is free.
    for r in range(len(nodes) + 1):
        for extra in combinations(nodes, r):
            s_side = {source, *extra}
            cap = sum(c for (u, v, c) in edges if u in s_side and v not in s_side)
            if cap < best:
                best = cap
    return best


# ----------------------------------------------------------------------------
# Public, domain-level result
# ----------------------------------------------------------------------------

@dataclass
class CutEdge:
    """One capacity-bearing edge on the reported minimum cut, identified back
    to the domain record it came from via `kind` ("demand" | "drain" | "outfall")
    and `ref` (catchment id / drain id / outfall junction id respectively)."""
    tail: str
    head: str
    capacity: float
    kind: str
    ref: str


@dataclass
class FloodCutResult:
    """The full Phase 2 answer for one (network, scenario) pair."""
    scenario_id: str
    max_flow_m3s: float
    iterations: int
    total_demand_m3s: float
    total_overflow_m3s: float
    catchment_demand_m3s: dict[str, float]
    catchment_delivered_m3s: dict[str, float]
    catchment_overflow_m3s: dict[str, float]
    outfall_capacity_m3s: dict[str, float]
    outfall_delivered_m3s: dict[str, float]
    drain_flow_m3s: dict[str, float]
    min_cut_edges: list[CutEdge]
    min_cut_capacity_m3s: float
    source_side_nodes: set[str]
    cross_check_value_m3s: float | None = None
    cross_check_agrees: bool | None = None
    note_on_overflow_uniqueness: str = (
        "Catchment-level overflow and outfall-level delivered flow are read off "
        "ONE maximum flow (this run's). The scenario TOTAL overflow is unique "
        "(every maximum flow has the same value), but if the bottleneck sits "
        "downstream of where two catchments' paths merge, a different maximum "
        "flow of equal total value could distribute the shortfall between them "
        "differently. See docs/ASSUMPTIONS.md A7 and maxflow.py section 5."
    )

    def summary_lines(self) -> list[str]:
        lines = [
            f"Scenario {self.scenario_id}: demand={self.total_demand_m3s:.3f} m3/s, "
            f"max flow (deliverable)={self.max_flow_m3s:.3f} m3/s, "
            f"overflow={self.total_overflow_m3s:.3f} m3/s "
            f"({self._pct():.1f}% of demand overflows)",
            f"Edmonds-Karp augmentations: {self.iterations}",
        ]
        if self.cross_check_value_m3s is not None:
            mark = "OK" if self.cross_check_agrees else "MISMATCH"
            lines.append(f"Cross-check vs NetworkX max flow: {self.cross_check_value_m3s:.6f} "
                         f"m3/s [{mark}]")
        lines.append(f"Min cut capacity: {self.min_cut_capacity_m3s:.3f} m3/s "
                      f"({len(self.min_cut_edges)} edge(s)):")
        for ce in sorted(self.min_cut_edges, key=lambda e: -e.capacity):
            lines.append(f"  [{ce.kind:7s}] {ce.ref:8s} {ce.tail} -> {ce.head} "
                          f"capacity={ce.capacity:.3f}")
        for cid, ov in sorted(self.catchment_overflow_m3s.items(), key=lambda kv: -kv[1]):
            if ov > _EPS:
                lines.append(f"  overflow at catchment {cid}: {ov:.3f} m3/s "
                             f"of {self.catchment_demand_m3s[cid]:.3f} m3/s demand")
        return lines

    def _pct(self) -> float:
        return (100.0 * self.total_overflow_m3s / self.total_demand_m3s
                if self.total_demand_m3s > _EPS else 0.0)


def analyze(net: DrainageNetwork, flow_network: FlowNetwork, *, cross_check: bool = True
            ) -> FloodCutResult:
    """Run the Phase 2 engine on an already-built `FlowNetwork` (see graph.py).

    Computes max flow with our own Edmonds-Karp, extracts a minimum cut from
    the final residual graph, reads off per-catchment and per-outfall flows,
    and (by default) cross-checks the max-flow *value* against NetworkX's
    default algorithm (`preflow_push`, a different algorithm family from
    Edmonds-Karp, run independently on the same graph) as an implementation
    sanity check. Disable with `cross_check=False` for speed on large graphs.
    """
    graph = flow_network.graph
    max_flow_value, adj, iterations = edmonds_karp(graph, SOURCE, SINK)
    source_side, cut_edges_internal = _min_cut_from_residual(adj, SOURCE)

    cut_edges = [CutEdge(tail=e.tail, head=e.to, capacity=e.capacity, kind=e.kind, ref=e.ref)
                 for e in cut_edges_internal]
    min_cut_capacity = sum(ce.capacity for ce in cut_edges)
    if abs(min_cut_capacity - max_flow_value) > max(1e-6, 1e-6 * max(1.0, max_flow_value)):
        # This would indicate a bug in this module (see module docstring, section 4):
        # the theorem guarantees exact equality, so this is not a "soft" sanity check.
        raise FloodCutError(
            f"internal error: min cut capacity ({min_cut_capacity}) != max flow "
            f"({max_flow_value}); max-flow/min-cut invariant violated")

    catchment_demand: dict[str, float] = {}
    catchment_delivered: dict[str, float] = {}
    catchment_overflow: dict[str, float] = {}
    for c in net.catchments.values():
        cn = catchment_node(c.id)
        demand_edge = next(e for e in adj[SOURCE] if e.is_original and e.to == cn)
        delivered = demand_edge.flow
        catchment_demand[c.id] = demand_edge.capacity
        catchment_delivered[c.id] = delivered
        catchment_overflow[c.id] = max(0.0, demand_edge.capacity - delivered)

    outfall_capacity: dict[str, float] = {}
    outfall_delivered: dict[str, float] = {}
    for o in net.outfalls.values():
        edge = next((e for e in adj[o.junction_id] if e.is_original and e.to == SINK), None)
        if edge is None:
            # Outfall junction has no outfall edge reachable in this build (shouldn't
            # happen given build_flow_network always adds one per Outfall record),
            # guarded defensively rather than assumed.
            outfall_capacity[o.junction_id] = 0.0
            outfall_delivered[o.junction_id] = 0.0
            continue
        outfall_capacity[o.junction_id] = edge.capacity
        outfall_delivered[o.junction_id] = edge.flow

    drain_flow: dict[str, float] = {}
    for d in net.drains.values():
        dn = drain_node(d.id)
        tail, _ = flow_network.drain_edge(d.id)
        edge = next(e for e in adj[tail] if e.is_original and e.to == dn)
        drain_flow[d.id] = edge.flow

    total_demand = sum(catchment_demand.values())
    total_overflow = total_demand - max_flow_value
    # Floating point: clamp away a possible -1e-13-style artefact.
    total_overflow = max(0.0, total_overflow) if abs(total_overflow) < 1e-6 else total_overflow

    cross_value: float | None = None
    cross_agrees: bool | None = None
    if cross_check:
        cross_value = float(nx.maximum_flow_value(graph, SOURCE, SINK))
        cross_agrees = abs(cross_value - max_flow_value) <= max(1e-6, 1e-6 * max(1.0, cross_value))

    return FloodCutResult(
        scenario_id=flow_network.scenario.id,
        max_flow_m3s=max_flow_value,
        iterations=iterations,
        total_demand_m3s=total_demand,
        total_overflow_m3s=total_overflow,
        catchment_demand_m3s=catchment_demand,
        catchment_delivered_m3s=catchment_delivered,
        catchment_overflow_m3s=catchment_overflow,
        outfall_capacity_m3s=outfall_capacity,
        outfall_delivered_m3s=outfall_delivered,
        drain_flow_m3s=drain_flow,
        min_cut_edges=cut_edges,
        min_cut_capacity_m3s=min_cut_capacity,
        source_side_nodes=source_side,
        cross_check_value_m3s=cross_value,
        cross_check_agrees=cross_agrees,
    )
