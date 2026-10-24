"""Cross-record validation of a DrainageNetwork.

Returns a list of Issues instead of raising, so callers can show every problem at once.
ERROR  = the network cannot be used for analysis.
WARNING = usable, but check it (e.g. a catchment that cannot reach any outfall).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .models import DrainageNetwork

# Rough bounding box of Greater Mumbai, used only to flag obviously wrong coordinates.
MUMBAI_BBOX = {"lat": (18.85, 19.30), "lon": (72.75, 73.05)}


@dataclass(frozen=True)
class Issue:
    severity: str  # "ERROR" | "WARNING"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == "ERROR"]


def validate_network(net: DrainageNetwork) -> list[Issue]:
    issues: list[Issue] = []

    def err(code: str, msg: str) -> None:
        issues.append(Issue("ERROR", code, msg))

    def warn(code: str, msg: str) -> None:
        issues.append(Issue("WARNING", code, msg))

    if not net.outfalls:
        err("E_NO_OUTFALL", "network has no outfalls; no water can leave")

    # Referential integrity
    for d in net.drains.values():
        for end in (d.from_junction, d.to_junction):
            if end not in net.junctions:
                err("E_DRAIN_ENDPOINT", f"drain {d.id} references unknown junction {end}")
    for c in net.catchments.values():
        if c.junction_id not in net.junctions:
            err("E_CATCHMENT_JUNCTION", f"catchment {c.id} references unknown junction {c.junction_id}")
    for o in net.outfalls.values():
        if o.junction_id not in net.junctions:
            err("E_OUTFALL_JUNCTION", f"outfall references unknown junction {o.junction_id}")
    for a in net.assets.values():
        if a.catchment_id not in net.catchments:
            err("E_ASSET_CATCHMENT", f"asset {a.id} references unknown catchment {a.catchment_id}")

    # Outfalls are sinks: nothing may leave them.
    for d in net.drains.values():
        if d.from_junction in net.outfalls:
            err("E_OUTFALL_HAS_OUTFLOW", f"drain {d.id} leaves outfall junction {d.from_junction}")

    # Structural checks on the simple directed graph of valid endpoints.
    g = nx.DiGraph()
    g.add_nodes_from(net.junctions)
    for d in net.drains.values():
        if d.from_junction in net.junctions and d.to_junction in net.junctions:
            g.add_edge(d.from_junction, d.to_junction)

    for jid in net.junctions:
        has_drain = g.degree(jid) > 0
        has_link = any(c.junction_id == jid for c in net.catchments.values()) or jid in net.outfalls
        if not has_drain and not has_link:
            warn("W_ISOLATED_JUNCTION", f"junction {jid} has no drains, catchment or outfall")

    outfall_ids = [j for j in net.outfalls if j in g]
    for c in net.catchments.values():
        if c.junction_id in g:
            reach = nx.descendants(g, c.junction_id) | {c.junction_id}
            if not reach.intersection(outfall_ids):
                warn("W_CATCHMENT_NO_PATH", f"catchment {c.id} cannot reach any outfall")

    if not nx.is_directed_acyclic_graph(g):
        warn("W_CYCLE", "drain graph contains a directed cycle (gravity drainage is normally acyclic)")

    lat_lo, lat_hi = MUMBAI_BBOX["lat"]
    lon_lo, lon_hi = MUMBAI_BBOX["lon"]
    for j in net.junctions.values():
        if not (lat_lo <= j.lat <= lat_hi and lon_lo <= j.lon <= lon_hi):
            warn("W_OUTSIDE_BBOX", f"junction {j.id} lies outside the Greater Mumbai bounding box")

    return issues
