import networkx as nx
import pytest

from floodcut.graph import (SINK, SOURCE, build_flow_network, build_physical_graph, catchment_node,
                            drain_node)
from floodcut.models import RainfallScenario

from conftest import make_network


def test_physical_graph_structure_and_parallel_drains(sim_network):
    g = build_physical_graph(sim_network)
    assert isinstance(g, nx.MultiDiGraph)
    assert g.number_of_nodes() == 14 and g.number_of_edges() == 14
    assert g.number_of_edges("J07", "J08") == 2                      # parallel pair preserved
    assert {k for _, _, k in g.edges("J07", keys=True)} == {"D08a", "D08b"}
    assert g.nodes["OUT1"]["is_outfall"] and not g.nodes["J01"]["is_outfall"]
    assert g["J05"]["J07"]["D05"]["effective_capacity"] == pytest.approx(1.8)   # 3.0 * 0.6
    assert nx.is_directed_acyclic_graph(nx.DiGraph(g))


def test_flow_network_encoding(sim_network, sim_scenarios):
    sc = sim_scenarios["design"]
    fn = build_flow_network(sim_network, sc)
    g = fn.graph
    kinds = [d["kind"] for _, _, d in g.edges(data=True)]
    assert kinds.count("demand") == 6 and kinds.count("catchment_link") == 6
    assert kinds.count("drain") == 14 and kinds.count("drain_link") == 14
    assert kinds.count("outfall") == 2
    # Demand edges carry rational-method runoff; total matches the sum.
    q_c1 = 0.8 * 50 * 12 / 360
    assert g[SOURCE][catchment_node("C1")]["capacity"] == pytest.approx(q_c1)
    assert fn.total_demand_m3s == pytest.approx(
        sum(c.runoff_m3s(50) for c in sim_network.catchments.values()))
    # Drain identity and effective capacity (incl. fully blocked drain = 0 capacity edge).
    u, v = fn.drain_edge("D08b")
    assert (u, v) == ("J07", drain_node("D08b")) and g[u][v]["capacity"] == 0.0
    assert g[fn.drain_edge("D05")[0]][drain_node("D05")]["capacity"] == pytest.approx(1.8)
    # Link edges have no capacity attribute (infinite in NetworkX).
    assert "capacity" not in g[drain_node("D01")]["J05"]
    assert fn.demand_edge("C2") == (SOURCE, "catchment:C2")


def test_tide_factor_scales_outfall_edges_only(sim_network):
    low = build_flow_network(sim_network, RainfallScenario("a", "a", 50, 1.0)).graph
    high = build_flow_network(sim_network, RainfallScenario("b", "b", 50, 0.5)).graph
    assert low["OUT1"][SINK]["capacity"] == pytest.approx(5.0)
    assert high["OUT1"][SINK]["capacity"] == pytest.approx(2.5)
    assert high["OUT2"][SINK]["capacity"] == pytest.approx(1.75)
    assert high[SOURCE]["catchment:C1"]["capacity"] == low[SOURCE]["catchment:C1"]["capacity"]


def test_unknown_drain_raises(sim_network, sim_scenarios):
    fn = build_flow_network(sim_network, sim_scenarios["light"])
    with pytest.raises(KeyError):
        fn.drain_edge("NOPE")


# --- Representation sanity checks on tiny hand-computed networks -------------------------
# These only confirm the *encoding* is faithful (series -> minimum, parallel -> sum).
# They are not the Phase 2 analysis engine.

def test_encoding_series_bottleneck_hand_computed():
    # demand: C=1, i=36 mm/h, A=100 ha -> 10 m3/s; drains in series 4 then 4 -> deliverable 4
    net = make_network(["A", "B", "T"], [("D1", "A", "B", 4.0), ("D2", "B", "T", 4.0)],
                       [("C1", "A", 100, 1.0)], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    assert fn.total_demand_m3s == pytest.approx(10.0)
    assert nx.maximum_flow_value(fn.graph, SOURCE, SINK) == pytest.approx(4.0)


def test_encoding_parallel_drains_hand_computed():
    # demand 5 m3/s (A=50 ha, i=36); two parallel drains of 2 each -> deliverable 4
    net = make_network(["A", "T"], [("D1", "A", "T", 2.0), ("D2", "A", "T", 2.0)],
                       [("C1", "A", 50, 1.0)], [("T", 100.0)])
    fn = build_flow_network(net, RainfallScenario("s", "s", 36, 1.0))
    assert fn.total_demand_m3s == pytest.approx(5.0)
    assert nx.maximum_flow_value(fn.graph, SOURCE, SINK) == pytest.approx(4.0)
