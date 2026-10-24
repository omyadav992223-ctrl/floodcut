"""Graph representations built from a DrainageNetwork.

1. build_physical_graph  -> nx.MultiDiGraph: the drainage network as it is on the ground
   (junction nodes, one edge per drain, parallel drains preserved).

2. build_flow_network    -> nx.DiGraph with super-source S* and super-sink T*, for one
   rainfall scenario. This is the *encoding* that Phase 2 max-flow runs on. Phase 1 only
   builds it; no flow is computed here.

Flow-network encoding (every item keeps its identity, parallel drains are supported):

    S* --[q_c]--> catchment:C --(inf)--> J_in --[cap_d]--> drain:D --(inf)--> J_out --[cap_o]--> T*

 - demand edge       S* -> catchment:C     capacity = rational-method runoff of catchment C
 - catchment link    catchment:C -> J      no capacity attribute (= infinite in NetworkX)
 - drain edge        U -> drain:D          capacity = effective capacity of drain D
 - drain link        drain:D -> V          no capacity attribute (= infinite)
 - outfall edge      J -> T*               capacity = outfall capacity x tide factor

Every edge carries `kind` and `ref` (the id of the originating record) so that a min cut
can later be mapped back to drains, catchments or outfalls. Capacity is on exactly one
edge per record; the node splitting for drains is why parallel drains need no special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .models import DrainageNetwork, RainfallScenario

SOURCE = "S*"
SINK = "T*"


def drain_node(drain_id: str) -> str:
    return f"drain:{drain_id}"


def catchment_node(catchment_id: str) -> str:
    return f"catchment:{catchment_id}"


def build_physical_graph(net: DrainageNetwork) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph(name=net.metadata.name, is_simulated=net.metadata.is_simulated)
    for j in net.junctions.values():
        g.add_node(j.id, name=j.name, lat=j.lat, lon=j.lon, is_outfall=j.id in net.outfalls)
    for d in net.drains.values():
        g.add_edge(d.from_junction, d.to_junction, key=d.id, drain_id=d.id,
                   design_capacity=d.design_capacity_m3s, capacity_factor=d.capacity_factor,
                   effective_capacity=d.effective_capacity_m3s, repair_cost=d.repair_cost,
                   kind=d.kind)
    return g


@dataclass
class FlowNetwork:
    graph: nx.DiGraph
    scenario: RainfallScenario
    total_demand_m3s: float

    def drain_edge(self, drain_id: str) -> tuple[str, str]:
        """The capacity-bearing edge (from_junction -> drain node) of a drain."""
        node = drain_node(drain_id)
        if node not in self.graph:
            raise KeyError(f"unknown drain {drain_id!r}")
        (upstream,) = self.graph.predecessors(node)
        return (upstream, node)

    def demand_edge(self, catchment_id: str) -> tuple[str, str]:
        return (SOURCE, catchment_node(catchment_id))


def build_flow_network(net: DrainageNetwork, scenario: RainfallScenario) -> FlowNetwork:
    g = nx.DiGraph(name=f"{net.metadata.name} | scenario={scenario.id}",
                   is_simulated=net.metadata.is_simulated)
    g.add_node(SOURCE, kind="source")
    g.add_node(SINK, kind="sink")
    for jid in net.junctions:
        g.add_node(jid, kind="junction", is_outfall=jid in net.outfalls)

    total = 0.0
    for c in net.catchments.values():
        q = c.runoff_m3s(scenario.intensity_mm_h)
        total += q
        cn = catchment_node(c.id)
        g.add_node(cn, kind="catchment")
        g.add_edge(SOURCE, cn, capacity=q, kind="demand", ref=c.id)
        g.add_edge(cn, c.junction_id, kind="catchment_link", ref=c.id)

    for d in net.drains.values():
        dn = drain_node(d.id)
        g.add_node(dn, kind="drain")
        g.add_edge(d.from_junction, dn, capacity=d.effective_capacity_m3s, kind="drain", ref=d.id)
        g.add_edge(dn, d.to_junction, kind="drain_link", ref=d.id)

    for o in net.outfalls.values():
        g.add_edge(o.junction_id, SINK, capacity=o.effective_capacity_m3s(scenario.tide_factor),
                   kind="outfall", ref=o.junction_id)

    return FlowNetwork(graph=g, scenario=scenario, total_demand_m3s=total)
