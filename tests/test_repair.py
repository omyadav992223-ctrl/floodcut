"""Tests for Phase 4 repair optimisation (src/floodcut/repair.py).

Exit-criteria checklist (ROADMAP.md Phase 4):
  [x] Exhaustive optimum beats or equals all baselines on every test network.
  [x] Greedy-failure example reproduced: network where greedy is suboptimal.
  [x] Interacting repairs: sub-additive and super-additive gain examples.
  [x] Zero-budget plan: no drains repaired, no cost, no reduction.
  [x] Full-budget plan (budget > total repair cost): all candidates repaired.
  [x] Costs and overflow reduction are consistent (base − repaired).
  [x] Exhaustive raises RepairOptError above MAX_EXHAUSTIVE_SUBSETS.
  [x] Candidate selection: only drains with capacity_factor < 1 are candidates.
  [x] Repair is non-destructive: original network is unchanged.
  [x] Bundled 14-drain network smoke test.
  [x] CLI smoke test for the ``repair`` subcommand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Imports from the package under test
# ---------------------------------------------------------------------------
from floodcut.repair import (
    RepairOptError,
    RepairPlan,
    OptimisationReport,
    _candidate_drains,
    _apply_repairs,
    _run,
    compare_strategies,
    exhaustive_repair,
    greedy_repair,
    cheapest_baseline,
    severity_baseline,
    ratio_baseline,
    random_baseline,
)
from floodcut.models import (
    Catchment, Drain, DrainageNetwork, Junction, NetworkMetadata, Outfall, RainfallScenario,
)
from floodcut.io import load_network, load_scenarios

# ---------------------------------------------------------------------------
# Test helpers (replicate conftest.make_network without importing pytest there)
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "data" / "simulated"

DESIGN = RainfallScenario("design", "Design storm", 50, 1.0)
LIGHT  = RainfallScenario("light",  "Light rain",   25, 1.0)


def make_network(junction_ids, drains, catchments, outfalls,
                 simulated=True) -> DrainageNetwork:
    net = DrainageNetwork(
        metadata=NetworkMetadata(name="test", is_simulated=simulated, source="test")
    )
    for j in junction_ids:
        net.junctions[j] = Junction(j, j, 19.0, 72.85)
    for d in drains:
        did, a, b, cap, *rest = d
        factor = rest[0] if rest else 1.0
        repair_cost = rest[1] if len(rest) > 1 else 0.0
        net.drains[did] = Drain(did, a, b, cap, factor, repair_cost)
    for cid, jid, area, c in catchments:
        net.catchments[cid] = Catchment(cid, cid, jid, area, c)
    for jid, cap in outfalls:
        net.outfalls[jid] = Outfall(jid, jid, cap)
    return net


def _linear_degraded():
    """A --D1(cap=4, factor=0.5, cost=10)--> B --D2(cap=4, factor=1.0, cost=10)--> OUT.

    Demand: 10 m³/s (always far exceeds capacity).
    Healthy max flow = 4 m³/s.  Base: D1 effective = 2 m³/s, max flow = 2, overflow = 8.
    Repairing D1 → max flow = 4, overflow = 6, reduction = 2.
    Repairing D2 alone → max flow still limited by D1=2, no gain.
    """
    return make_network(
        ["A", "B", "OUT"],
        [("D1", "A", "B", 4.0, 0.5, 10.0),
         ("D2", "B", "OUT", 4.0, 1.0, 10.0)],
        [("C1", "A", 100, 1.0)],
        [("OUT", 100.0)],
    )


def _two_path_degraded():
    """Two independent paths, each partially blocked.

    Path 1: A --D1(cap=3, factor=0.5, cost=5)--> B --D2(cap=3, factor=1, cost=5)--> OUT
    Path 2: A --D3(cap=2, factor=0.5, cost=4)--> C --D4(cap=2, factor=1, cost=4)--> OUT

    Base max flow = 1.5 + 1.0 = 2.5 m³/s.  Demand >> capacity → base overflow = demand − 2.5.
    Repair D1 alone: path 1 capacity 3, total flow = 3 + 1 = 4.  Δ = 1.5.
    Repair D3 alone: path 2 capacity 2, total flow = 1.5 + 2 = 3.5.  Δ = 1.0.
    Repair both: total flow = 5.  Δ = 2.5.  Sub-additive? 1.5+1.0 = 2.5 = combined. Exactly additive.
    """
    return make_network(
        ["A", "B", "C", "OUT"],
        [("D1", "A", "B", 3.0, 0.5, 5.0),
         ("D2", "B", "OUT", 3.0, 1.0, 5.0),
         ("D3", "A", "C", 2.0, 0.5, 4.0),
         ("D4", "C", "OUT", 2.0, 1.0, 4.0)],
        [("C1", "A", 100, 1.0)],
        [("OUT", 100.0)],
    )


def _shared_bottleneck_net():
    """Sub-additive repair example (shared downstream bottleneck).

    A --D1(cap=3,factor=0.5,cost=6)--> M --D_bottleneck(cap=3,factor=1)--> OUT
    A --D2(cap=3,factor=0.5,cost=6)--> M

    D1 and D2 both feed M through the same bottleneck (cap=3).
    Base: effective caps D1=1.5, D2=1.5 → flow through bottleneck = min(3, 3) = 3.
    Wait: M receives 1.5+1.5 = 3, but bottleneck cap = 3 → max flow = 3. overflow = demand-3.
    Repair D1 only: D1=3, D2=1.5 → inflow to M = 4.5, bottleneck = 3 → flow = 3. Gain = 0.
    Repair D2 only: same → gain = 0.
    Repair both: D1=3, D2=3 → inflow = 6, bottleneck = 3 → flow = 3. Combined gain = 0.

    Hmm, bottleneck already saturated at base. Let's adjust so repairs DO give gain but sub-additively.
    Use bottleneck cap=5:
    Base: 1.5+1.5 = 3 → flow = 3.
    Repair D1: 3+1.5 = 4.5 → flow = 4.5.  Gain = 1.5.
    Repair D2: 1.5+3 = 4.5 → flow = 4.5.  Gain = 1.5.
    Repair both: 3+3 = 6 > 5 → flow = 5.  Combined gain = 2.  Sum of marginals = 3. SUB-ADDITIVE.
    """
    return make_network(
        ["A", "M", "OUT"],
        [("D1", "A", "M", 3.0, 0.5, 6.0),
         ("D2", "A", "M", 3.0, 0.5, 6.0),
         ("DB", "M", "OUT", 5.0, 1.0, 0.0)],  # bottleneck, not degraded
        [("C1", "A", 100, 1.0)],
        [("OUT", 100.0)],
    )


def _greedy_suboptimal_net():
    """A network where greedy picks the wrong first drain and ends up suboptimal.

    Structure:
      A --D_big(cap=5, factor=0.1, cost=10)--> OUT   [effective cap = 0.5]
      A --D_a(cap=3, factor=0.5, cost=5)---> OUT     [effective cap = 1.5]
      A --D_b(cap=3, factor=0.5, cost=5)---> OUT     [effective cap = 1.5]

    Budget = 10.  Demand >> capacity so overflow = demand - sum(effective caps).

    At the base state:
      Repair D_big (cost=10): gain = 4.5, cost = 10.  Exactly uses budget.
      Repair D_a (cost=5):    gain = 1.5.
      Repair D_b (cost=5):    gain = 1.5.

    Greedy step 1: D_big has highest gain (4.5 > 1.5), picks D_big.
      → exhausts budget. Total reduction = 4.5.
    Exhaustive: D_a + D_b, cost=10, gain = 3.0.  OR D_big alone, gain=4.5.
      → D_big alone is actually better.  So greedy IS optimal here.

    Adjust: make D_big gain = 2.0 (factor 0.1 → but what cap?) and D_a+D_b gain = 3.0.
    Use D_big(cap=2.5, factor=0.1, cost=10): effective=0.25, repair → effective=2.5, gain=2.25.
    D_a(cap=3, factor=0.5, cost=5): effective=1.5, repair → effective=3.0, gain=1.5.
    D_b(cap=3, factor=0.5, cost=5): effective=1.5, repair → effective=3.0, gain=1.5.
    Greedy: picks D_a (gain 1.5, cheapest among 1.5-gain drains), remaining=5.
      Step 2: repair D_b (gain 1.5 from current state, cost=5) → total gain=3.0.
    Actually greedy marginal re-evaluation: after repairing D_a, D_b still gives 1.5 marginal.
    Alternative: D_big alone → gain=2.25.  3.0 > 2.25 so greedy (D_a+D_b) IS optimal too.

    Truly force greedy to be suboptimal with a serial interaction:
    A --D1(cap=3,factor=0.5,cost=8) --> B --D2(cap=3,factor=0.5,cost=8) --> OUT
    A --D3(cap=3,factor=1.0,cost=0) --> C --D4(cap=2,factor=0.5,cost=5) --> OUT

    Budget=13.  Base: path AB effective=1.5 (bottleneck at min(1.5,1.5)=1.5), path CD=2.
    Base overflow = demand - 3.5.
    Repair D1 (cost=8): path AB limited by D2=1.5 → flow=1.5, no change.  Gain=0.
    Repair D2 (cost=8): path AB limited by D1=1.5 → flow=1.5, no change.  Gain=0.
    Repair D4 (cost=5): path CD → 3 → 2 limited by design_cap of D4=2 → 2, no gain since already max.
    Hmm.  Let me use a cleaner setup.

    REAL greedy suboptimal example (classic "density vs value"):
    Single outfall, three drains each feeding it from three separate catchments.
    D_h(cap=4, factor=0.4, cost=10, repair gain=2.4): "high value"
    D_m1(cap=3, factor=0.5, cost=5, repair gain=1.5): "medium"
    D_m2(cap=3, factor=0.5, cost=5, repair gain=1.5): "medium"
    Budget=10.

    Greedy step 1: highest gain is D_h (2.4), cost=10. Exhausts budget.
    Exhaustive: D_m1+D_m2, gain=3.0 > 2.4. Greedy is SUBoptimal!
    """
    return make_network(
        ["A", "B", "C", "OUT"],
        [("D_h",  "A", "OUT", 4.0, 0.4, 10.0),   # repair gain 2.4
         ("D_m1", "B", "OUT", 3.0, 0.5, 5.0),    # repair gain 1.5
         ("D_m2", "C", "OUT", 3.0, 0.5, 5.0)],   # repair gain 1.5
        [("C_A", "A", 50, 1.0),
         ("C_B", "B", 50, 1.0),
         ("C_C", "C", 50, 1.0)],
        [("OUT", 100.0)],  # outfall large enough not to be the bottleneck
    )


# ===========================================================================
# 1. Candidate selection
# ===========================================================================

def test_candidate_selection_excludes_healthy_drains():
    net = _linear_degraded()
    # D1 factor=0.5 (candidate), D2 factor=1.0 (not a candidate)
    candidates = _candidate_drains(net)
    assert "D1" in candidates
    assert "D2" not in candidates


def test_candidate_selection_all_degraded():
    net = _two_path_degraded()
    candidates = _candidate_drains(net)
    # D1(0.5) and D3(0.5) are degraded; D2(1.0) and D4(1.0) are healthy
    assert set(candidates) == {"D1", "D3"}


def test_candidate_selection_fully_blocked():
    """Fully blocked drains (factor=0.0) are valid candidates."""
    net = make_network(
        ["A", "OUT"],
        [("D1", "A", "OUT", 3.0, 0.0, 5.0)],
        [("C1", "A", 10, 1.0)],
        [("OUT", 100.0)],
    )
    assert "D1" in _candidate_drains(net)


# ===========================================================================
# 2. _apply_repairs does not mutate the original
# ===========================================================================

def test_apply_repairs_non_destructive():
    net = _linear_degraded()
    original_factor = net.drains["D1"].capacity_factor
    repaired_net = _apply_repairs(net, ["D1"])
    assert net.drains["D1"].capacity_factor == original_factor  # original unchanged
    assert repaired_net.drains["D1"].capacity_factor == 1.0     # copy repaired
    assert repaired_net.drains["D2"].capacity_factor == net.drains["D2"].capacity_factor


# ===========================================================================
# 3. Zero-budget plan
# ===========================================================================

def test_zero_budget_no_repairs():
    net = _linear_degraded()
    base_ov = _run(net, DESIGN).total_overflow_m3s
    plan = exhaustive_repair(net, DESIGN, budget=0.0)
    assert plan.repaired_drain_ids == ()
    assert plan.total_cost == 0.0
    assert math.isclose(plan.total_overflow_m3s, base_ov, abs_tol=1e-9)
    assert math.isclose(plan.overflow_reduction_m3s, 0.0, abs_tol=1e-9)


def test_zero_budget_greedy():
    net = _linear_degraded()
    base_ov = _run(net, DESIGN).total_overflow_m3s
    plan = greedy_repair(net, DESIGN, budget=0.0)
    assert plan.repaired_drain_ids == ()
    assert math.isclose(plan.total_overflow_m3s, base_ov, abs_tol=1e-9)


# ===========================================================================
# 4. Full-budget plan: all candidates repaired
# ===========================================================================

def test_full_budget_repairs_all_candidates():
    net = _two_path_degraded()
    # D1 cost=5, D3 cost=4 → total = 9
    plan = exhaustive_repair(net, DESIGN, budget=100.0)
    assert set(plan.repaired_drain_ids) == {"D1", "D3"}


# ===========================================================================
# 5. Linear network: only the upstream bottleneck benefits
# ===========================================================================

def test_linear_repair_d1_reduces_overflow():
    net = _linear_degraded()
    base_ov = _run(net, DESIGN).total_overflow_m3s
    plan_d1 = _run(net, DESIGN, ["D1"]).total_overflow_m3s
    plan_d2 = _run(net, DESIGN, ["D2"]).total_overflow_m3s

    # Repairing the upstream bottleneck (D1) should reduce overflow
    assert plan_d1 < base_ov - 1e-9

    # Repairing D2 alone cannot help: D1 is still the bottleneck
    assert math.isclose(plan_d2, base_ov, abs_tol=1e-9)


def test_linear_exhaustive_picks_d1():
    net = _linear_degraded()
    plan = exhaustive_repair(net, DESIGN, budget=10.0)
    assert "D1" in plan.repaired_drain_ids
    assert "D2" not in plan.repaired_drain_ids


# ===========================================================================
# 6. Sub-additive repair (shared downstream bottleneck)
# ===========================================================================

def test_shared_bottleneck_repair_gains_are_subadditive():
    net = _shared_bottleneck_net()
    base_ov = _run(net, DESIGN).total_overflow_m3s

    gain_d1 = base_ov - _run(net, DESIGN, ["D1"]).total_overflow_m3s
    gain_d2 = base_ov - _run(net, DESIGN, ["D2"]).total_overflow_m3s
    gain_both = base_ov - _run(net, DESIGN, ["D1", "D2"]).total_overflow_m3s

    # Each drain individually opens the path further
    assert gain_d1 > 1e-9
    assert gain_d2 > 1e-9

    # Combined gain is strictly less than the sum (sub-additive)
    assert gain_both < gain_d1 + gain_d2 - 1e-9, (
        f"Expected sub-additivity: {gain_both:.4f} < {gain_d1 + gain_d2:.4f}"
    )


# ===========================================================================
# 7. Greedy suboptimality example (the key Phase 4 exit criterion)
# ===========================================================================

def test_greedy_is_suboptimal_on_density_example():
    """Classic counter-example: two cheap medium-value drains beat one high-value drain."""
    net = _greedy_suboptimal_net()
    base_ov = _run(net, DESIGN).total_overflow_m3s

    # Verify the expected marginal gains at base state.
    gain_h  = base_ov - _run(net, DESIGN, ["D_h"]).total_overflow_m3s
    gain_m1 = base_ov - _run(net, DESIGN, ["D_m1"]).total_overflow_m3s
    gain_m2 = base_ov - _run(net, DESIGN, ["D_m2"]).total_overflow_m3s

    # D_h has the highest individual gain (greedy will pick it first)
    assert gain_h > gain_m1
    assert gain_h > gain_m2

    # D_m1 + D_m2 combined beats D_h alone (independent paths — additive gains)
    gain_both_m = base_ov - _run(net, DESIGN, ["D_m1", "D_m2"]).total_overflow_m3s
    assert gain_both_m > gain_h + 1e-9, (
        f"Test setup: D_m1+D_m2 gain {gain_both_m:.4f} must exceed D_h gain {gain_h:.4f}"
    )

    # Budget=10: greedy picks D_h (exhausts budget), exhaustive picks D_m1+D_m2.
    plan_greedy = greedy_repair(net, DESIGN, budget=10.0)
    plan_opt    = exhaustive_repair(net, DESIGN, budget=10.0)

    assert "D_h" in plan_greedy.repaired_drain_ids, "Greedy should pick D_h first"
    assert set(plan_opt.repaired_drain_ids) == {"D_m1", "D_m2"}, (
        f"Optimum should pick D_m1+D_m2, got {plan_opt.repaired_drain_ids}"
    )

    # The optimum beats greedy
    assert plan_opt.overflow_reduction_m3s > plan_greedy.overflow_reduction_m3s + 1e-9, (
        f"Optimum reduction {plan_opt.overflow_reduction_m3s:.4f} must exceed "
        f"greedy {plan_greedy.overflow_reduction_m3s:.4f}"
    )


# ===========================================================================
# 8. Exhaustive always beats or equals all baselines
# ===========================================================================

def _assert_exhaustive_optimal(net, scenario, budget, label=""):
    """Helper: exhaustive reduction >= every strategy's reduction."""
    report = compare_strategies(net, scenario, budget, run_exhaustive=True)
    opt = report.exhaustive_optimal
    assert opt is not None, f"{label}: exhaustive should have run"
    for name, plan in report.plans.items():
        if name == "exhaustive":
            continue
        assert opt.overflow_reduction_m3s >= plan.overflow_reduction_m3s - 1e-9, (
            f"{label}: exhaustive ({opt.overflow_reduction_m3s:.4f}) < "
            f"{name} ({plan.overflow_reduction_m3s:.4f})"
        )


def test_exhaustive_optimal_linear_low_budget():
    _assert_exhaustive_optimal(_linear_degraded(), DESIGN, budget=10.0, label="linear/low")


def test_exhaustive_optimal_linear_high_budget():
    _assert_exhaustive_optimal(_linear_degraded(), DESIGN, budget=50.0, label="linear/high")


def test_exhaustive_optimal_two_path():
    _assert_exhaustive_optimal(_two_path_degraded(), DESIGN, budget=5.0, label="two-path/5")


def test_exhaustive_optimal_two_path_tight_budget():
    _assert_exhaustive_optimal(_two_path_degraded(), DESIGN, budget=4.0, label="two-path/4")


def test_exhaustive_optimal_shared_bottleneck():
    _assert_exhaustive_optimal(_shared_bottleneck_net(), DESIGN, budget=6.0, label="bottleneck/6")


def test_exhaustive_optimal_density_example():
    _assert_exhaustive_optimal(_greedy_suboptimal_net(), DESIGN, budget=10.0, label="density/10")


# ===========================================================================
# 9. RepairPlan consistency checks
# ===========================================================================

def test_plan_cost_equals_sum_of_drain_costs():
    net = _two_path_degraded()
    plan = exhaustive_repair(net, DESIGN, budget=9.0)
    expected_cost = sum(net.drains[did].repair_cost for did in plan.repaired_drain_ids)
    assert math.isclose(plan.total_cost, expected_cost, abs_tol=1e-9)


def test_plan_reduction_equals_base_minus_repaired():
    net = _linear_degraded()
    base_ov = _run(net, DESIGN).total_overflow_m3s
    plan = exhaustive_repair(net, DESIGN, budget=10.0, base_overflow=base_ov)
    expected_red = max(0.0, base_ov - plan.total_overflow_m3s)
    assert math.isclose(plan.overflow_reduction_m3s, expected_red, abs_tol=1e-9)


def test_plan_cost_within_budget():
    net = _shared_bottleneck_net()
    budget = 6.0
    for fn in (greedy_repair, exhaustive_repair):
        plan = fn(net, DESIGN, budget)
        assert plan.total_cost <= budget + 1e-9, f"{fn.__name__}: cost {plan.total_cost} > {budget}"


# ===========================================================================
# 10. Baselines are feasible (stay within budget)
# ===========================================================================

def test_baselines_feasible():
    net = _greedy_suboptimal_net()
    budget = 10.0
    for fn in (severity_baseline, cheapest_baseline, ratio_baseline):
        plan = fn(net, DESIGN, budget)
        assert plan.total_cost <= budget + 1e-9, (
            f"{fn.__name__}: cost {plan.total_cost:.2f} exceeds budget {budget}"
        )


def test_random_baseline_feasible():
    net = _greedy_suboptimal_net()
    budget = 10.0
    plan = random_baseline(net, DESIGN, budget, seed=0)
    assert plan.total_cost <= budget + 1e-9


# ===========================================================================
# 11. Exhaustive raises on too-large search spaces
# ===========================================================================

def test_exhaustive_raises_on_large_search_space(monkeypatch):
    """Patch MAX_SUBSETS to a tiny value to force the safety check."""
    import floodcut.repair as rep_mod
    monkeypatch.setattr(rep_mod, "_MAX_EXHAUSTIVE_SUBSETS", 3)
    net = _two_path_degraded()  # 2 candidates → 4 subsets > 3
    with pytest.raises(RepairOptError, match="limit"):
        exhaustive_repair(net, DESIGN, budget=100.0)


# ===========================================================================
# 12. compare_strategies gap reporting
# ===========================================================================

def test_gap_is_zero_when_strategy_matches_optimum():
    """Cheapest repairs: both greedy and exhaustive pick D3 (cost=4) with budget=4."""
    net = _two_path_degraded()
    report = compare_strategies(net, DESIGN, budget=4.0)
    # Gap for exhaustive itself is 0
    assert report.gap("exhaustive") == 0.0


def test_gap_is_positive_when_strategy_is_suboptimal():
    """On the density example with budget=10, greedy should have a positive gap."""
    net = _greedy_suboptimal_net()
    report = compare_strategies(net, DESIGN, budget=10.0)
    assert report.gap("greedy") is not None
    assert report.gap("greedy") > 1e-9


def test_gap_none_when_exhaustive_not_run():
    net = _greedy_suboptimal_net()
    report = compare_strategies(net, DESIGN, budget=10.0, run_exhaustive=False)
    assert report.exhaustive_optimal is None
    assert report.gap("greedy") is None


# ===========================================================================
# 13. Greedy re-evaluates in current repaired state
# ===========================================================================

def test_greedy_reeval_linear_picks_upstream_first():
    """In the linear net, greedy must recognise D2 gives 0 gain even though it's cheap."""
    net = _linear_degraded()
    base_ov = _run(net, DESIGN).total_overflow_m3s
    plan = greedy_repair(net, DESIGN, budget=10.0)
    # Should repair D1 (the upstream bottleneck)
    assert "D1" in plan.repaired_drain_ids
    assert plan.overflow_reduction_m3s > 1e-9


# ===========================================================================
# 14. super-additive example (two drains opening a new series path together)
# ===========================================================================

def test_super_additive_series_drains():
    """Two drains in series: repairing either alone gains nothing; both together open the path.

    A --D1(cap=3, factor=0.0, cost=5)--> B --D2(cap=3, factor=0.0, cost=5)--> OUT
    D3(cap=3, factor=1.0, cost=0) gives a baseline path: A --D3--> OUT (cap=3).

    Wait, that changes the setup. Let's use:
    Catchment only at A.
    A --D1(factor=0, cost=5)--> B --D2(factor=0, cost=5)--> OUT
    No other path.
    Base: max flow = 0 (both blocked).  Demand >> 0 → base overflow = demand.
    Repair D1 alone: D1 now cap=3 but D2 still blocked → path still dead.  Gain = 0.
    Repair D2 alone: same → gain = 0.
    Repair both: full path open → max flow = 3. Gain = 3.  SUPER-ADDITIVE (0+0 < 3).
    """
    net = make_network(
        ["A", "B", "OUT"],
        [("D1", "A", "B", 3.0, 0.0, 5.0),   # fully blocked
         ("D2", "B", "OUT", 3.0, 0.0, 5.0)], # fully blocked
        [("C1", "A", 100, 1.0)],
        [("OUT", 100.0)],
    )
    base_ov = _run(net, DESIGN).total_overflow_m3s
    gain_d1 = base_ov - _run(net, DESIGN, ["D1"]).total_overflow_m3s
    gain_d2 = base_ov - _run(net, DESIGN, ["D2"]).total_overflow_m3s
    gain_both = base_ov - _run(net, DESIGN, ["D1", "D2"]).total_overflow_m3s

    assert math.isclose(gain_d1, 0.0, abs_tol=1e-9), "Repairing D1 alone must give 0 gain"
    assert math.isclose(gain_d2, 0.0, abs_tol=1e-9), "Repairing D2 alone must give 0 gain"
    assert gain_both > 1e-9, "Repairing both must open the path"
    assert gain_both > gain_d1 + gain_d2 + 1e-9, "Must be super-additive"


# ===========================================================================
# 15. Bundled 14-drain simulated network smoke test
# ===========================================================================

def test_bundled_network_compare_strategies():
    """Smoke test: compare_strategies runs without error on the bundled network."""
    net = load_network(DATA / "mumbai_sim_network.json")
    scenarios = load_scenarios(DATA / "rainfall_scenarios.json")
    scenario = scenarios.get("design") or next(iter(scenarios.values()))

    # Degraded drains: D04 (factor=0.8), D05 (factor=0.6), D08b (factor=0.0), D11 (factor=0.5)
    candidates = _candidate_drains(net)
    assert len(candidates) >= 4

    # Budget covering roughly two repairs
    budget = 50.0
    report = compare_strategies(net, scenario, budget=budget, run_exhaustive=True)

    assert report.base_overflow_m3s >= 0.0
    for name, plan in report.plans.items():
        assert plan.total_cost <= budget + 1e-9, f"{name} exceeds budget"
        assert plan.overflow_reduction_m3s >= -1e-9


def test_bundled_network_greedy_reduces_overflow():
    net = load_network(DATA / "mumbai_sim_network.json")
    scenarios = load_scenarios(DATA / "rainfall_scenarios.json")
    sc = scenarios.get("design") or next(iter(scenarios.values()))

    base_ov = _run(net, sc).total_overflow_m3s
    plan = greedy_repair(net, sc, budget=30.0)
    # Should not increase overflow
    assert plan.total_overflow_m3s <= base_ov + 1e-9


# ===========================================================================
# 16. CLI smoke test
# ===========================================================================

def test_cli_repair_command(capsys):
    """The repair subcommand should produce output and exit 0."""
    from floodcut.cli import main
    rc = main(["repair", "--budget", "30"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "greedy" in captured.out.lower() or "repair" in captured.out.lower()


def test_cli_repair_no_exhaustive(capsys):
    """--no-exhaustive flag should skip exhaustive search (no 'exhaustive' row in table)."""
    from floodcut.cli import main
    rc = main(["repair", "--budget", "30", "--no-exhaustive"])
    assert rc == 0
    captured = capsys.readouterr()
    # The exhaustive plan row starts with "exhaustive" as the strategy label in the table.
    # The summary footer mentions it was not run, but there should be no plan row for it.
    lines = captured.out.splitlines()
    table_rows = [l for l in lines if l.startswith("exhaustive")]
    assert table_rows == [], f"Unexpected exhaustive table row: {table_rows}"


def test_cli_repair_scenario(capsys):
    """repair with --scenario moderate should work."""
    from floodcut.cli import main
    rc = main(["repair", "--budget", "40", "--scenario", "moderate"])
    assert rc == 0
