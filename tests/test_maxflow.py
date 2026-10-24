"""Tests for the Phase 2 engine (src/floodcut/maxflow.py).

Three independent lines of evidence are used, per the roadmap's Phase 2 exit
criteria ("max flow = min cut verified by brute-force cut enumeration on small
graphs; hand-computed examples pass"):

1. Hand-computed examples (manually verifiable arithmetic, checked in comments).
2. Brute-force min-cut enumeration (a definition-level check, independent of
   any max-flow algorithm) on small graphs, including random ones.
3. Cross-checks against NetworkX's own max-flow (a different algorithm family,
   preflow-push, run independently) on both the bundled dataset and random graphs.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from floodcut.graph import SINK, SOURCE, build_flow_network, catchment_node
from floodcut.maxflow import (FloodCutError, analyze, brute_force_min_cut, edmonds_karp)
from floodcut.models import RainfallScenario

from conftest import make_network


# --------------------------------------------------------------------------
# 1. Hand-computed examples
# --------------------------------------------------------------------------

def test_hand_computed_series_bottleneck():
    # C=1, i=36mm/h, A=100ha -> demand = 1*36*100/360 = 10 m3/s.
    # Series drains A->B (cap 4) -> T (cap 4): the smaller of two capacities in
    # series bounds the path, so max flow = 4, overflow = 10 - 4 = 6.
    net = make_network(["A", "B", "T"], [("D1", "A", "B", 4.0), ("D2", "B", "T", 4.0)],
                       [("C1", "A", 100, 1.0)], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.max_flow_m3s == pytest.approx(4.0)
    assert result.total_demand_m3s == pytest.approx(10.0)
    assert result.total_overflow_m3s == pytest.approx(6.0)
    assert result.catchment_overflow_m3s["C1"] == pytest.approx(6.0)
    assert result.drain_flow_m3s == {"D1": pytest.approx(4.0), "D2": pytest.approx(4.0)}
    assert result.cross_check_agrees is True


def test_hand_computed_parallel_drains_sum_capacity():
    # demand 5 m3/s (A=50ha, i=36 -> 1*36*50/360=5); two parallel drains A->T of
    # capacity 2 each: parallel capacities ADD, so max flow = 4 (not 2), overflow = 1.
    net = make_network(["A", "T"], [("D1", "A", "T", 2.0), ("D2", "A", "T", 2.0)],
                       [("C1", "A", 50, 1.0)], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.max_flow_m3s == pytest.approx(4.0)
    assert result.total_overflow_m3s == pytest.approx(1.0)
    assert sum(result.drain_flow_m3s.values()) == pytest.approx(4.0)
    # Both parallel drains should be saturated (each is a min-cut edge).
    cut_refs = {ce.ref for ce in result.min_cut_edges}
    assert cut_refs == {"D1", "D2"}


def test_hand_computed_diamond_bottleneck_in_the_middle():
    # A -> {B, C} -> D -> T. Two parallel upstream branches (cap 3 each -> 6 total
    # available) feed a single downstream drain of cap 2, which is the true
    # bottleneck: max flow = 2 regardless of the generous upstream capacity.
    net = make_network(
        ["A", "B", "C", "D", "T"],
        [("D1", "A", "B", 3.0), ("D2", "A", "C", 3.0), ("D3", "B", "D", 3.0),
         ("D4", "C", "D", 3.0), ("D5", "D", "T", 2.0)],
        [("C1", "A", 100, 1.0)], [("T", 100.0)])
    # demand = 1*36*100/360 = 10, way above the bottleneck.
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.max_flow_m3s == pytest.approx(2.0)
    assert {ce.ref for ce in result.min_cut_edges} == {"D5"}
    assert brute_force_min_cut(fn.graph, SOURCE, SINK) == pytest.approx(2.0)


def test_sim_network_design_scenario_hand_verified(sim_network, sim_scenarios):
    """Full manual trace of the bundled 14-junction network at the 'design'
    scenario (50 mm/h, tide 1.0), reproduced in docs/MAXFLOW.md section 'worked
    example'. Two independent sub-basins (OUT1 via trunk J07, OUT2 via J10/J11)
    are joined by one cross-link drain D07 (J06 -> J10):

      OUT1 side: J05 gets C1+C2 = 1.3333+0.875 = 2.2083 m3/s of demand, but
      D05 (J05->J07) caps that at 1.8. J07's own output is capped at
      D08a+D08b = 3.0+0 = 3.0 (D08b is fully blocked, capacity_factor=0), so
      1.2 more is pulled from J06 via D06, leaving J06 with 2.9306-1.2=1.7306
      spare, of which D07 (cap 1.2) carries 1.2 onward to J10.
      -> OUT1 delivers exactly 3.0 m3/s (its own trunk bottleneck).

      OUT2 side: J10 receives C5's 1.5556 plus the 1.2 handed across by D07
      (2.7556 total) but D11 (J10->J11) caps that at 1.6. J11 then also gets
      C6's full 0.7778 via D13, for 1.6+0.7778 = 2.3778, which clears D12 and
      the outfall (both capacity 3.5) unconstrained.
      -> OUT2 delivers exactly 2.3778 m3/s.

      Total deliverable = 3.0 + 2.3778 = 5.3778 m3/s.
      Total demand = 7.4722 m3/s (sum of all six catchments' rational-method
      runoff). Overflow = 7.4722 - 5.3778 = 2.0944 m3/s.
    """
    sc = sim_scenarios["design"]
    fn = build_flow_network(sim_network, sc)
    result = analyze(sim_network, fn)

    assert result.total_demand_m3s == pytest.approx(7.4722, abs=5e-4)
    assert result.max_flow_m3s == pytest.approx(5.3778, abs=5e-4)
    assert result.total_overflow_m3s == pytest.approx(2.0944, abs=5e-4)
    assert result.outfall_delivered_m3s["OUT1"] == pytest.approx(3.0, abs=5e-4)
    assert result.outfall_delivered_m3s["OUT2"] == pytest.approx(2.3778, abs=5e-4)
    assert result.cross_check_agrees is True
    # Bookkeeping: delivered must equal demand minus overflow, per catchment and in total.
    for cid, demand in result.catchment_demand_m3s.items():
        assert result.catchment_delivered_m3s[cid] + result.catchment_overflow_m3s[cid] \
            == pytest.approx(demand, abs=1e-9)
    assert sum(result.catchment_delivered_m3s.values()) == pytest.approx(result.max_flow_m3s)
    assert sum(result.catchment_overflow_m3s.values()) == pytest.approx(result.total_overflow_m3s)


def test_sim_network_light_scenario_has_no_overflow(sim_network, sim_scenarios):
    # Design-storm capacities comfortably exceed light-rain demand network-wide.
    fn = build_flow_network(sim_network, sim_scenarios["light"])
    result = analyze(sim_network, fn)
    assert result.total_overflow_m3s == pytest.approx(0.0, abs=1e-6)
    assert result.max_flow_m3s == pytest.approx(result.total_demand_m3s, abs=1e-6)


# --------------------------------------------------------------------------
# 2. Edge cases: disconnected graphs, zero capacity, invalid input
# --------------------------------------------------------------------------

def test_zero_capacity_blocked_drain_in_sim_data_does_not_break_engine(sim_network, sim_scenarios):
    # D08b has capacity_factor=0.0 in the bundled dataset -> effective capacity 0.
    fn = build_flow_network(sim_network, sim_scenarios["design"])
    result = analyze(sim_network, fn)
    assert result.drain_flow_m3s["D08b"] == 0.0
    # A zero-capacity edge is trivially "saturated" and may legally appear on
    # the reported min cut (residual capacity 0 from the start).
    assert all(ce.capacity >= 0 for ce in result.min_cut_edges)


def test_disconnected_catchment_overflows_completely():
    # Junction B has a drain but it dead-ends at X, which has no path to the
    # only outfall T. C1's entire demand must overflow; the rest of the graph
    # is unaffected.
    net = make_network(
        ["A", "B", "X", "T"],
        [("D1", "A", "T", 10.0), ("D2", "B", "X", 10.0)],
        [("C1", "A", 10, 0.5), ("C2", "B", 10, 0.5)],
        [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.catchment_overflow_m3s["C2"] == pytest.approx(result.catchment_demand_m3s["C2"])
    assert result.catchment_overflow_m3s["C1"] == pytest.approx(0.0, abs=1e-9)


def test_wholly_disconnected_source_and_sink_gives_zero_flow():
    # Two components that never meet: max flow must be exactly 0, not an error.
    net = make_network(["A", "T"], [], [("C1", "A", 10, 0.5)], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.max_flow_m3s == 0.0
    assert result.total_overflow_m3s == pytest.approx(result.total_demand_m3s)
    assert result.min_cut_edges == []  # no finite-capacity edge crosses the reachable set
    assert result.min_cut_capacity_m3s == 0.0


def test_no_catchments_or_zero_demand_gives_zero_flow_not_an_error():
    net = make_network(["A", "T"], [("D1", "A", "T", 5.0)], [], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    result = analyze(net, fn)
    assert result.max_flow_m3s == 0.0
    assert result.total_demand_m3s == 0.0
    assert result.total_overflow_m3s == 0.0


def test_source_missing_raises():
    g = nx.DiGraph()
    g.add_edge("a", "b", capacity=1.0)
    with pytest.raises(FloodCutError, match="not a node"):
        edmonds_karp(g, "S*", "b")


def test_sink_missing_raises():
    g = nx.DiGraph()
    g.add_edge("a", "b", capacity=1.0)
    with pytest.raises(FloodCutError, match="not a node"):
        edmonds_karp(g, "a", "T*")


def test_source_equals_sink_raises():
    g = nx.DiGraph()
    g.add_node("a")
    with pytest.raises(FloodCutError, match="must be different"):
        edmonds_karp(g, "a", "a")


def test_negative_capacity_rejected():
    g = nx.DiGraph()
    g.add_edge("s", "t", capacity=-1.0)
    with pytest.raises(FloodCutError, match=">= 0"):
        edmonds_karp(g, "s", "t")


def test_nan_capacity_rejected():
    g = nx.DiGraph()
    g.add_edge("s", "t", capacity=float("nan"))
    with pytest.raises(FloodCutError, match=">= 0"):
        edmonds_karp(g, "s", "t")


def test_non_numeric_capacity_rejected():
    g = nx.DiGraph()
    g.add_edge("s", "t", capacity="a lot")
    with pytest.raises(FloodCutError, match="must be a number"):
        edmonds_karp(g, "s", "t")


def test_bool_capacity_rejected():
    # bool is a subclass of int in Python; must be explicitly rejected like
    # Drain.design_capacity_m3s already is in models.py.
    g = nx.DiGraph()
    g.add_edge("s", "t", capacity=True)
    with pytest.raises(FloodCutError, match="must be a number"):
        edmonds_karp(g, "s", "t")


def test_infinite_capacity_edge_handled_as_infinite():
    # No 'capacity' attribute at all = infinite, matching NetworkX's own convention
    # (graph.py relies on this for catchment_link / drain_link edges).
    g = nx.DiGraph()
    g.add_edge("s", "m", capacity=3.0)
    g.add_edge("m", "t")  # infinite
    value, _, _ = edmonds_karp(g, "s", "t")
    assert value == pytest.approx(3.0)


def test_brute_force_min_cut_refuses_large_graphs():
    g = nx.DiGraph()
    nodes = [f"n{i}" for i in range(25)]
    for a, b in zip(nodes, nodes[1:]):
        g.add_edge(a, b, capacity=1.0)
    with pytest.raises(FloodCutError, match="exponential"):
        brute_force_min_cut(g, nodes[0], nodes[-1])


# --------------------------------------------------------------------------
# 3. Independent validation against NetworkX and against brute-force min cut
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id", ["light", "design", "heavy", "extreme"])
def test_all_bundled_scenarios_cross_check_against_networkx(sim_network, sim_scenarios, scenario_id):
    fn = build_flow_network(sim_network, sim_scenarios[scenario_id])
    result = analyze(sim_network, fn)
    nx_value = nx.maximum_flow_value(fn.graph, SOURCE, SINK)
    assert result.max_flow_m3s == pytest.approx(nx_value, abs=1e-6)
    assert result.cross_check_agrees is True
    # Our reported min cut capacity must equal both values too (the theorem).
    assert result.min_cut_capacity_m3s == pytest.approx(nx_value, abs=1e-6)


def _random_capacitated_digraph(rng: random.Random, n: int, edge_prob: float,
                                 allow_zero_capacity: bool) -> nx.DiGraph:
    """Random small DiGraph on nodes 0..n-1, source=0, sink=n-1. May be
    disconnected or contain directed cycles (both must be handled correctly)."""
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < edge_prob:
                lo = 0 if allow_zero_capacity else 1
                cap = rng.randint(lo, 10)
                g.add_edge(i, j, capacity=float(cap))
    return g


@pytest.mark.parametrize("seed", range(25))
def test_random_small_graphs_match_networkx_and_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 7)
    g = _random_capacitated_digraph(rng, n, edge_prob=0.4, allow_zero_capacity=True)
    source, sink = 0, n - 1

    ours, _, iterations = edmonds_karp(g, source, sink)
    theirs = nx.maximum_flow_value(g, source, sink)
    brute = brute_force_min_cut(g, source, sink)

    assert ours == pytest.approx(theirs, abs=1e-9)
    assert ours == pytest.approx(brute, abs=1e-9)
    # Edmonds-Karp's classical bound: at most O(V*E) augmentations.
    assert iterations <= n * g.number_of_edges() + 1


@pytest.mark.parametrize("seed", range(10))
def test_random_graphs_with_cycles_match_networkx(seed):
    # edge_prob high enough that directed cycles are likely; Edmonds-Karp must
    # remain correct (and must terminate) in their presence.
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 6)
    g = _random_capacitated_digraph(rng, n, edge_prob=0.6, allow_zero_capacity=True)
    assert not nx.is_directed_acyclic_graph(g) or True  # just documents intent; not all seeds cycle
    source, sink = 0, n - 1
    ours, _, _ = edmonds_karp(g, source, sink)
    theirs = nx.maximum_flow_value(g, source, sink)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_random_graph_with_antiparallel_edges_handled_independently():
    # u->v and v->u both present with different capacities: a naive
    # "merge residual into the reverse edge" implementation would corrupt one
    # of these; our per-edge reverse pointer must keep them independent.
    g = nx.DiGraph()
    g.add_edge("s", "a", capacity=5.0)
    g.add_edge("a", "b", capacity=4.0)
    g.add_edge("b", "a", capacity=3.0)  # anti-parallel to a->b
    g.add_edge("b", "t", capacity=5.0)
    value, _, _ = edmonds_karp(g, "s", "t")
    assert value == pytest.approx(4.0)  # bounded by a->b, the true bottleneck
    assert value == pytest.approx(nx.maximum_flow_value(g, "s", "t"))


def test_min_cut_never_includes_an_infinite_capacity_edge(sim_network, sim_scenarios):
    fn = build_flow_network(sim_network, sim_scenarios["extreme"])
    result = analyze(sim_network, fn)
    assert all(ce.kind in ("demand", "drain", "outfall") for ce in result.min_cut_edges)


def test_max_flow_min_cut_invariant_holds_for_every_scenario(sim_network, sim_scenarios):
    for sc in sim_scenarios.values():
        fn = build_flow_network(sim_network, sc)
        result = analyze(sim_network, fn)
        assert result.min_cut_capacity_m3s == pytest.approx(result.max_flow_m3s, abs=1e-6)


def test_cli_maxflow_smoke(capsys):
    from floodcut.cli import main
    rc = main(["maxflow", "--scenario", "design"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Min cut capacity" in out
    assert "Cross-check vs NetworkX" in out
