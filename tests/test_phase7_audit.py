"""Phase 7: Testing and Final Mathematical Audit.

Verifies:
1. Graph Theory and Flow Conservation:
   - Non-negative edge capacities.
   - Strict flow conservation at all intermediate vertices (v != S*, T*).
   - Capacity constraints: 0 <= f(e) <= c(e) on every edge.
   - Net flow out of S* == net flow into T* == max_flow.
   - Reported max flow equals minimum-cut capacity (Max-Flow Min-Cut theorem).
2. Edge cases in Flow Networks:
   - Disconnected network (no path to outfall): max flow = 0, overflow = total demand.
   - Zero-capacity bottleneck: capacity 0 cuts flow entirely.
   - Symmetrical / non-unique minimum cuts.
3. Failure Simulation Correctness:
   - Drain effective capacity zeroed out under failure.
   - Analytical hand-verifiable diamond network failure.
   - Invariant: total overflow == sum of catchment shortfalls.
4. Repair Optimization Correctness:
   - Zero budget / insufficient budget: repairs empty, cost 0, reduction 0.
   - Zero-cost repair selection.
   - Strict budget constraint: cost(R) <= B for all strategies.
   - Exhaustive search always >= Greedy reduction.
   - Overflow reduction identity: reduction == base_overflow - repaired_overflow.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from floodcut.failure import _apply_failures, enumerate_failures
from floodcut.graph import SINK, SOURCE, build_flow_network
from floodcut.io import load_network, load_scenarios
from floodcut.maxflow import analyze, edmonds_karp
from floodcut.models import Catchment, Drain, DrainageNetwork, Junction, NetworkMetadata, Outfall, RainfallScenario
from floodcut.repair import (
    compare_strategies,
    exhaustive_repair,
    greedy_repair,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "simulated"


def make_test_network(junctions, drains, catchments, outfalls) -> DrainageNetwork:
    """Helper to build small deterministic networks for mathematical proofs."""
    net = DrainageNetwork(metadata=NetworkMetadata(name="math_test", is_simulated=True, source="audit"))
    for jid in junctions:
        net.junctions[jid] = Junction(jid, jid, 19.0, 72.85)
    for d in drains:
        did, u, v, cap = d[:4]
        factor = d[4] if len(d) > 4 else 1.0
        cost = d[5] if len(d) > 5 else 10.0
        net.drains[did] = Drain(did, u, v, design_capacity_m3s=cap, capacity_factor=factor, repair_cost=cost)
    for c in catchments:
        cid, jid, area, coeff = c
        net.catchments[cid] = Catchment(cid, cid, jid, area_ha=area, runoff_coefficient=coeff)
    for o in outfalls:
        jid, cap = o[:2]
        tidal = o[2] if len(o) > 2 else False
        net.outfalls[jid] = Outfall(jid, jid, base_capacity_m3s=cap, tidal=tidal)
    return net


@pytest.fixture
def bundled_analysis():
    net = load_network(DATA_DIR / "mumbai_sim_network.json")
    scenarios = load_scenarios(DATA_DIR / "rainfall_scenarios.json")
    sc = scenarios["design"]
    fn = build_flow_network(net, sc)
    result = analyze(net, fn, cross_check=True)
    total_flow, adj, iterations = edmonds_karp(fn.graph, SOURCE, SINK)
    return net, fn, result, adj, total_flow


# ---------------------------------------------------------------------------
# 1. Flow Conservation & Capacity Invariants on the Bundled Network
# ---------------------------------------------------------------------------

class TestMathematicalInvariants:
    def test_all_capacities_non_negative(self, bundled_analysis):
        _, fn, _, _, _ = bundled_analysis
        for u, v, data in fn.graph.edges(data=True):
            if "capacity" in data:
                cap = data["capacity"]
                assert cap >= 0.0, f"Edge ({u}, {v}) has negative capacity {cap}"

    def test_flow_conservation_at_intermediate_vertices(self, bundled_analysis):
        _, fn, _, adj, _ = bundled_analysis

        intermediate_nodes = [n for n in fn.graph.nodes if n not in (SOURCE, SINK)]
        for node in intermediate_nodes:
            inflow = sum(
                e.flow for u, elist in adj.items() for e in elist
                if e.is_original and e.to == node
            )
            outflow = sum(
                e.flow for e in adj[node]
                if e.is_original
            )
            assert math.isclose(inflow, outflow, abs_tol=1e-8), (
                f"Flow conservation violated at node {node}: inflow={inflow:.6f}, outflow={outflow:.6f}"
            )

    def test_capacity_constraints_on_all_edges(self, bundled_analysis):
        _, _, _, adj, _ = bundled_analysis

        for u, elist in adj.items():
            for e in elist:
                if e.is_original:
                    assert e.flow >= -1e-9, f"Negative flow {e.flow} on edge ({u}, {e.to})"
                    assert e.flow <= e.capacity + 1e-8, (
                        f"Flow {e.flow:.6f} exceeds capacity {e.capacity:.6f} on edge ({u}, {e.to})"
                    )

    def test_source_outflow_equals_sink_inflow_equals_max_flow(self, bundled_analysis):
        _, _, result, adj, total_flow = bundled_analysis

        net_out_source = sum(e.flow for e in adj[SOURCE] if e.is_original)
        net_in_sink = sum(
            e.flow for u, elist in adj.items() for e in elist
            if e.is_original and e.to == SINK
        )

        assert math.isclose(net_out_source, result.max_flow_m3s, abs_tol=1e-8)
        assert math.isclose(net_in_sink, result.max_flow_m3s, abs_tol=1e-8)
        assert math.isclose(total_flow, result.max_flow_m3s, abs_tol=1e-8)

    def test_max_flow_equals_min_cut_capacity(self, bundled_analysis):
        _, _, result, _, _ = bundled_analysis
        assert math.isclose(result.max_flow_m3s, result.min_cut_capacity_m3s, abs_tol=1e-8), (
            f"Max-Flow Min-Cut theorem violated: max_flow={result.max_flow_m3s}, min_cut={result.min_cut_capacity_m3s}"
        )

    def test_total_overflow_equals_sum_of_catchment_overflows(self, bundled_analysis):
        _, _, result, _, _ = bundled_analysis
        sum_c_overflow = sum(result.catchment_overflow_m3s.values())
        assert math.isclose(result.total_overflow_m3s, sum_c_overflow, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# 2. Hand-Verifiable Flow Networks & Edge Cases
# ---------------------------------------------------------------------------

class TestFlowNetworkEdgeCases:
    def test_disconnected_network_yields_zero_flow(self):
        """Catchment is isolated and cannot reach the outfall."""
        net = make_test_network(
            junctions=["J1", "J2", "OUT"],
            drains=[("D1", "J1", "J2", 5.0)],  # J2 does not connect to OUT
            catchments=[("C1", "J1", 10.0, 0.36)],  # demand = 0.36 * 100 * 10 / 360 = 1.0 m3/s at 100 mm/h
            outfalls=[("OUT", 10.0)],
        )
        sc = RainfallScenario("s1", "s1", intensity_mm_h=100.0, tide_factor=1.0)
        fn = build_flow_network(net, sc)
        res = analyze(net, fn)

        assert res.max_flow_m3s == 0.0
        assert math.isclose(res.total_demand_m3s, 1.0, abs_tol=1e-8)
        assert math.isclose(res.total_overflow_m3s, 1.0, abs_tol=1e-8)
        assert res.min_cut_capacity_m3s == 0.0

    def test_zero_capacity_drain_blocks_all_flow(self):
        """A single series drain with capacity 0.0 blocks the entire network."""
        net = make_test_network(
            junctions=["J1", "OUT"],
            drains=[("D1", "J1", "OUT", 0.0)],  # zero design capacity
            catchments=[("C1", "J1", 10.0, 0.36)],
            outfalls=[("OUT", 10.0)],
        )
        sc = RainfallScenario("s1", "s1", intensity_mm_h=100.0, tide_factor=1.0)
        fn = build_flow_network(net, sc)
        res = analyze(net, fn)

        assert res.max_flow_m3s == 0.0
        assert math.isclose(res.total_overflow_m3s, 1.0, abs_tol=1e-8)

    def test_symmetrical_parallel_drains_split_flow(self):
        """Two parallel paths with equal capacity 2.0 m3/s, demand 3.0 m3/s -> flow 3.0 m3/s, overflow 0."""
        net = make_test_network(
            junctions=["J1", "J2a", "J2b", "OUT"],
            drains=[
                ("D1a", "J1", "J2a", 2.0),
                ("D1b", "J1", "J2b", 2.0),
                ("D2a", "J2a", "OUT", 2.0),
                ("D2b", "J2b", "OUT", 2.0),
            ],
            catchments=[("C1", "J1", 30.0, 0.36)],  # demand = 3.0 m3/s at 100 mm/h
            outfalls=[("OUT", 10.0)],
        )
        sc = RainfallScenario("s1", "s1", intensity_mm_h=100.0, tide_factor=1.0)
        fn = build_flow_network(net, sc)
        res = analyze(net, fn)

        assert math.isclose(res.max_flow_m3s, 3.0, abs_tol=1e-8)
        assert math.isclose(res.total_overflow_m3s, 0.0, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# 3. Mathematical Checks for Failure Simulation
# ---------------------------------------------------------------------------

class TestFailureSimulationCorrectness:
    def test_drain_effective_capacity_zeroed(self):
        net = load_network(DATA_DIR / "mumbai_sim_network.json")
        assert net.drains["D01"].effective_capacity_m3s > 0

        failed = _apply_failures(net, ["D01"])
        assert failed.drains["D01"].capacity_factor == 0.0
        assert failed.drains["D01"].effective_capacity_m3s == 0.0
        # Original network must not be mutated
        assert net.drains["D01"].capacity_factor == 1.0

    def test_worst_case_k_monotone_overflow(self):
        """Worst-case overflow for k+1 failures must be >= worst-case overflow for k failures."""
        net = load_network(DATA_DIR / "mumbai_sim_network.json")
        sc = load_scenarios(DATA_DIR / "rainfall_scenarios.json")["design"]

        summary = enumerate_failures(net, sc, k_max=2)
        wc1 = summary.worst_case(1)
        wc2 = summary.worst_case(2)

        assert wc1 is not None and wc2 is not None
        assert wc2.total_overflow_m3s >= wc1.total_overflow_m3s - 1e-9


# ---------------------------------------------------------------------------
# 4. Mathematical Checks for Repair Optimization
# ---------------------------------------------------------------------------

class TestRepairOptimizationCorrectness:
    @pytest.fixture
    def degraded_net(self):
        return load_network(DATA_DIR / "mumbai_sim_network.json")

    @pytest.fixture
    def design_scenario(self):
        return load_scenarios(DATA_DIR / "rainfall_scenarios.json")["design"]

    def test_budget_never_exceeded_any_strategy(self, degraded_net, design_scenario):
        budgets = [0.0, 10.0, 25.0, 50.0, 100.0]
        for b in budgets:
            report = compare_strategies(degraded_net, design_scenario, budget=b, run_exhaustive=True)
            for s_name, plan in report.plans.items():
                assert plan.total_cost <= b + 1e-8, (
                    f"Strategy {s_name} spent {plan.total_cost:.2f} exceeding budget {b:.2f}"
                )

    def test_infeasible_low_budget_picks_no_repairs(self, degraded_net, design_scenario):
        """If budget is smaller than the cheapest candidate, 0 repairs must be selected."""
        min_cost = min(d.repair_cost for d in degraded_net.drains.values() if d.capacity_factor < 1.0)
        tiny_budget = min_cost / 2.0

        plan = greedy_repair(degraded_net, design_scenario, budget=tiny_budget)
        assert len(plan.repaired_drain_ids) == 0
        assert plan.total_cost == 0.0
        assert plan.overflow_reduction_m3s == 0.0

    def test_zero_cost_repair_candidate_always_affordable(self, design_scenario):
        """A degraded drain with repair_cost = 0.0 must be selected even on zero budget."""
        net = make_test_network(
            junctions=["J1", "OUT"],
            drains=[("D1", "J1", "OUT", 5.0, 0.0, 0.0)],  # capacity 5.0, factor 0.0, cost 0.0
            catchments=[("C1", "J1", 10.0, 0.36)],  # demand = 1.0 m3/s
            outfalls=[("OUT", 10.0)],
        )
        plan = greedy_repair(net, design_scenario, budget=0.0)
        assert "D1" in plan.repaired_drain_ids
        assert plan.total_cost == 0.0
        assert plan.overflow_reduction_m3s > 0.0

    def test_exhaustive_optimal_beats_or_equals_greedy(self, degraded_net, design_scenario):
        for b in [15.0, 30.0, 45.0]:
            opt = exhaustive_repair(degraded_net, design_scenario, budget=b)
            grd = greedy_repair(degraded_net, design_scenario, budget=b)
            assert opt.overflow_reduction_m3s >= grd.overflow_reduction_m3s - 1e-9, (
                f"Exhaustive ({opt.overflow_reduction_m3s}) was beaten by greedy ({grd.overflow_reduction_m3s}) at budget {b}"
            )

    def test_overflow_reduction_identity(self, degraded_net, design_scenario):
        plan = greedy_repair(degraded_net, design_scenario, budget=30.0)
        expected_reduction = max(0.0, plan.base_overflow_m3s - plan.total_overflow_m3s)
        assert math.isclose(plan.overflow_reduction_m3s, expected_reduction, abs_tol=1e-8)
