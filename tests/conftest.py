from pathlib import Path

import pytest

from floodcut.io import load_network, load_scenarios
from floodcut.models import (Catchment, Drain, DrainageNetwork, Junction, NetworkMetadata, Outfall)

DATA = Path(__file__).resolve().parents[1] / "data" / "simulated"


@pytest.fixture(scope="session")
def sim_network():
    return load_network(DATA / "mumbai_sim_network.json")


@pytest.fixture(scope="session")
def sim_scenarios():
    return load_scenarios(DATA / "rainfall_scenarios.json")


def make_network(junction_ids, drains, catchments, outfalls, simulated=True) -> DrainageNetwork:
    """Tiny hand-built networks for tests. drains: (id, from, to, cap[, factor])."""
    net = DrainageNetwork(metadata=NetworkMetadata(name="test", is_simulated=simulated, source="test"))
    for j in junction_ids:
        net.junctions[j] = Junction(j, j, 19.0, 72.85)
    for d in drains:
        did, a, b, cap, *rest = d
        net.drains[did] = Drain(did, a, b, cap, *(rest[:1] or [1.0]))
    for cid, jid, area, c in catchments:
        net.catchments[cid] = Catchment(cid, cid, jid, area, c)
    for jid, cap in outfalls:
        net.outfalls[jid] = Outfall(jid, jid, cap)
    return net
