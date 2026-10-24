"""Phase 4: budget-constrained drain repair optimisation.

Given a drainage network in its current (degraded) state and a repair budget,
choose which drains to restore to their *design* capacity so as to reduce total
overflow as much as possible.

-------------------------------------------------------------------------------
1.  What "repair" means here
-------------------------------------------------------------------------------
Repairing drain d restores its ``capacity_factor`` to 1.0, so its effective
capacity becomes ``design_capacity_m3s`` (its healthy value).  We model this
by replacing d's ``capacity_factor`` with 1.0 before building the flow network
and re-running Phase 2.  The cost of that repair is ``d.repair_cost``.

Only drains with ``capacity_factor < 1.0`` are *candidates* for repair — there
is nothing to restore in an already-healthy drain.  Drains with
``capacity_factor == 0.0`` (fully blocked) are valid candidates; repairing them
is often the highest-value action.

A *repair plan* is a subset R of candidates.  Its total cost is
Σ repair_cost(d) for d in R; it is *feasible* if that cost does not exceed the
budget B.  Its *overflow reduction* is:

    Δ(R) = base_overflow – overflow(repaired network with R applied)

We maximise Δ(R) subject to cost(R) ≤ B.

-------------------------------------------------------------------------------
2.  Interaction effects
-------------------------------------------------------------------------------
Repairs interact because the drainage network is a flow network.  Repairing
two drains that share a downstream bottleneck can yield less than the sum of
their individual gains (sub-additivity).  Repairing two drains that together
open a new path can yield more than the sum (super-additivity).

The *greedy* algorithm (§3) ignores interactions at each step and can therefore
be suboptimal.  The *exhaustive* search (§4) evaluates every feasible subset
and is always optimal.

-------------------------------------------------------------------------------
3.  Greedy algorithm (marginal-gain greedy)
-------------------------------------------------------------------------------
Starting from the unrepaired (base) network state:

  1. Evaluate the marginal overflow reduction Δ(d) from repairing each
     candidate drain d *in the current repaired state* (not the base state).
  2. Pick the candidate with the highest Δ(d) that still fits in the remaining
     budget.  Break ties by lowest repair_cost (best value), then by drain id.
  3. Apply that repair, deduct its cost, and repeat from step 1.
  4. Stop when no candidate fits in the remaining budget, or every candidate
     has been repaired.

This is a classic greedy for submodular maximisation: it has a (1 – 1/e) ≈
0.63 approximation guarantee for monotone submodular functions.  Overflow
reduction under flow-network repair is in general not provably submodular, but
the greedy heuristic is fast and often optimal in practice.

Note that step 1 re-evaluates all remaining candidates from the *current*
repaired state, not from the base state.  This means the algorithm accounts
for interactions: it does not simply sort by marginal gain at the start and
follow that ranking blindly.

-------------------------------------------------------------------------------
4.  Exhaustive search (exact optimum for small candidate sets)
-------------------------------------------------------------------------------
Enumerate every subset of candidates whose total cost does not exceed the
budget, compute the overflow for each, and return the best.

Complexity: 2^n max-flow runs, where n = |candidates|.  For the bundled 14-
drain network (n ≤ 13 or fewer at any realistic capacity_factor distribution)
this is fast; for networks with many degraded drains and large budgets the
caller should prefer greedy or set a candidate limit.

A hard ceiling of 2^18 = 262 144 subsets is enforced; above that the function
raises ``RepairOptError`` rather than silently running for an unbounded time.

-------------------------------------------------------------------------------
5.  Baselines
-------------------------------------------------------------------------------
Four simple baselines are provided for comparison:

* **severity** – repair in decreasing order of current overflow contribution
  (largest marginal overflow first at base state), filling budget greedily.
* **cheapest** – repair cheapest candidates first.
* **ratio** – repair by decreasing overflow-reduction-per-cost-unit at base
  state (greedy fractional-knapsack heuristic, applied integrally).
* **random** – randomly sample feasible subsets ``n_trials`` times, return best.

All baselines are deterministic except *random*.

-------------------------------------------------------------------------------
6.  Optimality-gap report
-------------------------------------------------------------------------------
``compare_strategies`` runs all strategies and reports each plan's cost,
overflow reduction, and gap from the exhaustive optimum (when it was computed).
The gap is (optimum − strategy) / optimum, so a gap of 0 means the strategy
found the optimum.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from .graph import build_flow_network
from .maxflow import FloodCutResult, analyze
from .models import Drain, DrainageNetwork, RainfallScenario

_MAX_EXHAUSTIVE_SUBSETS = 2 ** 18  # hard ceiling to prevent runaway enumeration


class RepairOptError(ValueError):
    """Raised when repair optimisation cannot be safely performed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RepairPlan:
    """One repair plan: the drains chosen, cost, and resulting overflow."""
    repaired_drain_ids: tuple[str, ...]  # sorted
    total_cost: float
    result: FloodCutResult              # FloodCutResult from the repaired network
    base_overflow_m3s: float            # baseline before any repair
    strategy: str                       # human-readable label

    @property
    def total_overflow_m3s(self) -> float:
        return self.result.total_overflow_m3s

    @property
    def overflow_reduction_m3s(self) -> float:
        """How much overflow this plan removes relative to the base network."""
        return max(0.0, self.base_overflow_m3s - self.total_overflow_m3s)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Strategy: {self.strategy}",
            f"  Repaired drains: {list(self.repaired_drain_ids) if self.repaired_drain_ids else '(none)'}",
            f"  Total cost: {self.total_cost:.2f}",
            f"  Overflow after repair: {self.total_overflow_m3s:.4f} m3/s",
            f"  Overflow reduction: {self.overflow_reduction_m3s:.4f} m3/s  "
            f"({self._pct():.1f}% of base)",
        ]
        return lines

    def _pct(self) -> float:
        if self.base_overflow_m3s < 1e-9:
            return 0.0
        return 100.0 * self.overflow_reduction_m3s / self.base_overflow_m3s


@dataclass
class OptimisationReport:
    """Full comparison of all strategies against each other and the optimum."""
    scenario_id: str
    budget: float
    base_overflow_m3s: float
    candidate_drain_ids: tuple[str, ...]
    plans: dict[str, RepairPlan]          # strategy name → plan
    exhaustive_optimal: RepairPlan | None  # None if exhaustive was not run / too large

    def gap(self, strategy: str) -> float | None:
        """Fractional gap from the exhaustive optimum for a strategy (0 = optimal).

        Returns None if exhaustive was not run.
        """
        if self.exhaustive_optimal is None:
            return None
        opt_reduction = self.exhaustive_optimal.overflow_reduction_m3s
        if opt_reduction < 1e-9:
            return 0.0
        plan_reduction = self.plans[strategy].overflow_reduction_m3s
        return max(0.0, (opt_reduction - plan_reduction) / opt_reduction)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Repair optimisation: scenario={self.scenario_id}  budget={self.budget:.2f}",
            f"Base overflow: {self.base_overflow_m3s:.4f} m3/s  "
            f"candidates: {list(self.candidate_drain_ids)}",
        ]
        opt_red = (self.exhaustive_optimal.overflow_reduction_m3s
                   if self.exhaustive_optimal else None)
        lines.append(
            f"{'Strategy':<16}  {'Cost':>8}  {'Reduction m3/s':>16}  "
            f"{'Residual m3/s':>15}  {'Gap %':>7}  Drains"
        )
        lines.append("-" * 90)
        for name, plan in self.plans.items():
            gap = self.gap(name)
            gap_str = f"{100 * gap:6.1f}" if gap is not None else "     ?"
            lines.append(
                f"{name:<16}  {plan.total_cost:8.2f}  {plan.overflow_reduction_m3s:16.4f}  "
                f"{plan.total_overflow_m3s:15.4f}  {gap_str}  "
                f"{list(plan.repaired_drain_ids)}"
            )
        if self.exhaustive_optimal is not None:
            lines.append(
                f"\nOptimum (exhaustive): {list(self.exhaustive_optimal.repaired_drain_ids)}  "
                f"cost={self.exhaustive_optimal.total_cost:.2f}  "
                f"reduction={self.exhaustive_optimal.overflow_reduction_m3s:.4f} m3/s"
            )
        else:
            lines.append(
                "\n(Exhaustive search not run — candidate set too large or not requested.)"
            )
        return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _candidate_drains(net: DrainageNetwork) -> tuple[str, ...]:
    """Sorted ids of drains that can be repaired (capacity_factor < 1.0)."""
    return tuple(sorted(d.id for d in net.drains.values() if d.capacity_factor < 1.0))


def _apply_repairs(net: DrainageNetwork, repair_ids: Sequence[str]) -> DrainageNetwork:
    """Return a new DrainageNetwork with the specified drains restored to factor=1.0."""
    new_drains = dict(net.drains)
    for did in repair_ids:
        d = new_drains[did]
        new_drains[did] = Drain(
            id=d.id,
            from_junction=d.from_junction,
            to_junction=d.to_junction,
            design_capacity_m3s=d.design_capacity_m3s,
            capacity_factor=1.0,   # fully restored
            repair_cost=d.repair_cost,
            length_m=d.length_m,
            kind=d.kind,
        )
    return DrainageNetwork(
        metadata=net.metadata,
        junctions=net.junctions,
        drains=new_drains,
        catchments=net.catchments,
        outfalls=net.outfalls,
        assets=net.assets,
    )


def _run(net: DrainageNetwork, scenario: RainfallScenario,
         repair_ids: Sequence[str] = ()) -> FloodCutResult:
    """Run the max-flow engine on the network with the given drains repaired."""
    repaired_net = _apply_repairs(net, repair_ids) if repair_ids else net
    fn = build_flow_network(repaired_net, scenario)
    return analyze(repaired_net, fn, cross_check=False)


def _make_plan(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    repair_ids: Sequence[str],
    base_overflow: float,
    strategy: str,
) -> RepairPlan:
    result = _run(net, scenario, repair_ids)
    cost = sum(net.drains[did].repair_cost for did in repair_ids)
    return RepairPlan(
        repaired_drain_ids=tuple(sorted(repair_ids)),
        total_cost=cost,
        result=result,
        base_overflow_m3s=base_overflow,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def greedy_repair(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
) -> RepairPlan:
    """Marginal-gain greedy repair (see module docstring §3).

    Re-evaluates candidates from the current repaired state at each step,
    so interactions between repairs are accounted for.

    Parameters
    ----------
    net:
        Base (degraded) network.
    scenario:
        Rainfall/tide scenario.
    budget:
        Maximum total repair cost.
    base_overflow:
        Pre-computed base overflow (avoids one redundant max-flow call if
        the caller already has it).

    Returns
    -------
    RepairPlan with strategy label ``"greedy"``.
    """
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = set(_candidate_drains(net))
    remaining_budget = budget
    chosen: list[str] = []

    # State: the network with all repairs chosen so far applied.
    current_net = net
    current_overflow = base_overflow

    while True:
        # Affordable candidates (not yet chosen, within remaining budget).
        affordable = [
            did for did in candidates
            if net.drains[did].repair_cost <= remaining_budget
        ]
        if not affordable:
            break

        # Evaluate marginal gain of each affordable candidate in current state.
        best_did: str | None = None
        best_gain = -math.inf
        best_cost = math.inf

        for did in affordable:
            trial_net = _apply_repairs(current_net, [did])
            fn = build_flow_network(trial_net, scenario)
            trial_result = analyze(trial_net, fn, cross_check=False)
            gain = current_overflow - trial_result.total_overflow_m3s
            # Tie-break: highest gain → lowest cost → lexicographic id.
            if (gain > best_gain + 1e-9
                    or (abs(gain - best_gain) <= 1e-9
                        and (net.drains[did].repair_cost < best_cost
                             or (net.drains[did].repair_cost == best_cost
                                 and (best_did is None or did < best_did))))):
                best_gain = gain
                best_did = did
                best_cost = net.drains[did].repair_cost

        if best_did is None:
            break  # all affordable candidates give no improvement

        # Accept the best candidate.
        chosen.append(best_did)
        candidates.discard(best_did)
        remaining_budget -= net.drains[best_did].repair_cost
        current_net = _apply_repairs(current_net, [best_did])
        current_overflow -= best_gain

    return _make_plan(net, scenario, chosen, base_overflow, "greedy")


def exhaustive_repair(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
) -> RepairPlan:
    """Exact optimal repair by enumerating all feasible subsets (see §4).

    Raises
    ------
    RepairOptError
        If the number of feasible subsets exceeds ``_MAX_EXHAUSTIVE_SUBSETS``
        to prevent unbounded runtime.

    Returns
    -------
    RepairPlan with strategy label ``"exhaustive"``.
    """
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = _candidate_drains(net)
    n = len(candidates)

    # Count feasible subsets (cheap pre-check before running max-flow).
    feasible_count = sum(
        1
        for k in range(n + 1)
        for subset in combinations(candidates, k)
        if sum(net.drains[did].repair_cost for did in subset) <= budget
    )
    if feasible_count > _MAX_EXHAUSTIVE_SUBSETS:
        raise RepairOptError(
            f"Exhaustive search would evaluate {feasible_count} subsets "
            f"(limit {_MAX_EXHAUSTIVE_SUBSETS}). "
            f"Use greedy or reduce the candidate set / budget."
        )

    best_plan: RepairPlan | None = None

    for k in range(n + 1):
        for subset in combinations(candidates, k):
            cost = sum(net.drains[did].repair_cost for did in subset)
            if cost > budget:
                continue
            result = _run(net, scenario, subset)
            plan = RepairPlan(
                repaired_drain_ids=tuple(sorted(subset)),
                total_cost=cost,
                result=result,
                base_overflow_m3s=base_overflow,
                strategy="exhaustive",
            )
            if (best_plan is None
                    or plan.overflow_reduction_m3s > best_plan.overflow_reduction_m3s + 1e-9
                    or (abs(plan.overflow_reduction_m3s
                            - best_plan.overflow_reduction_m3s) <= 1e-9
                        and plan.total_cost < best_plan.total_cost)):
                best_plan = plan

    if best_plan is None:
        # Budget is zero or negative; return the no-repair plan.
        result = _run(net, scenario)
        best_plan = RepairPlan((), 0.0, result, base_overflow, "exhaustive")

    return best_plan


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def _greedy_from_ranking(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    base_overflow: float,
    ranked_candidates: Sequence[str],
    strategy: str,
) -> RepairPlan:
    """Fill budget greedily following a pre-computed candidate ranking.

    Unlike ``greedy_repair``, this does NOT re-evaluate gains from the current
    repaired state — it follows the initial ranking and accepts in order.
    """
    chosen: list[str] = []
    remaining = budget
    for did in ranked_candidates:
        cost = net.drains[did].repair_cost
        if cost <= remaining:
            chosen.append(did)
            remaining -= cost
    return _make_plan(net, scenario, chosen, base_overflow, strategy)


def severity_baseline(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
) -> RepairPlan:
    """Repair by decreasing base-state overflow contribution (most severe first).

    'Severity' of drain d = overflow(fail {d}) − base_overflow (the marginal
    overflow from losing d *given the base degraded state*, not a healthy one).
    Drains are ranked once from the base state; no re-evaluation after each
    choice (unlike ``greedy_repair``).
    """
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = _candidate_drains(net)

    # Severity: how much extra overflow does the base network suffer because
    # each candidate is degraded?  Proxy: gain from repairing it in isolation.
    marginal_gain: dict[str, float] = {}
    for did in candidates:
        trial = _run(net, scenario, [did])
        marginal_gain[did] = base_overflow - trial.total_overflow_m3s

    ranked = sorted(candidates,
                    key=lambda d: (-marginal_gain[d],
                                   net.drains[d].repair_cost,
                                   d))
    return _greedy_from_ranking(net, scenario, budget, base_overflow, ranked, "severity")


def cheapest_baseline(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
) -> RepairPlan:
    """Repair cheapest candidates first (ignores overflow impact)."""
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = _candidate_drains(net)
    ranked = sorted(candidates,
                    key=lambda d: (net.drains[d].repair_cost, d))
    return _greedy_from_ranking(net, scenario, budget, base_overflow, ranked, "cheapest")


def ratio_baseline(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
) -> RepairPlan:
    """Repair by decreasing overflow-reduction-per-cost-unit at base state.

    This is the fractional knapsack greedy applied integrally (items are
    indivisible drains, not fractions of drains).
    """
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = _candidate_drains(net)

    marginal_gain: dict[str, float] = {}
    for did in candidates:
        trial = _run(net, scenario, [did])
        marginal_gain[did] = base_overflow - trial.total_overflow_m3s

    def _ratio(did: str) -> float:
        cost = net.drains[did].repair_cost
        if cost < 1e-9:
            return math.inf if marginal_gain[did] > 0 else 0.0
        return marginal_gain[did] / cost

    ranked = sorted(candidates, key=lambda d: (-_ratio(d), net.drains[d].repair_cost, d))
    return _greedy_from_ranking(net, scenario, budget, base_overflow, ranked, "ratio")


def random_baseline(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    base_overflow: float | None = None,
    n_trials: int = 200,
    seed: int | None = 42,
) -> RepairPlan:
    """Best plan found by randomly sampling feasible subsets.

    Parameters
    ----------
    n_trials:
        Number of random feasible subsets to evaluate.
    seed:
        RNG seed for reproducibility (default 42).
    """
    if base_overflow is None:
        base_overflow = _run(net, scenario).total_overflow_m3s

    candidates = list(_candidate_drains(net))
    rng = random.Random(seed)
    best_plan: RepairPlan | None = None

    for _ in range(n_trials):
        # Build a random feasible subset: shuffle candidates, add while budget allows.
        rng.shuffle(candidates)
        subset: list[str] = []
        remaining = budget
        for did in candidates:
            if net.drains[did].repair_cost <= remaining:
                subset.append(did)
                remaining -= net.drains[did].repair_cost

        result = _run(net, scenario, subset)
        cost = sum(net.drains[did].repair_cost for did in subset)
        plan = RepairPlan(
            repaired_drain_ids=tuple(sorted(subset)),
            total_cost=cost,
            result=result,
            base_overflow_m3s=base_overflow,
            strategy="random",
        )
        if (best_plan is None
                or plan.overflow_reduction_m3s > best_plan.overflow_reduction_m3s + 1e-9):
            best_plan = plan

    if best_plan is None:
        result = _run(net, scenario)
        best_plan = RepairPlan((), 0.0, result, base_overflow, "random")

    return best_plan


# ---------------------------------------------------------------------------
# Combined comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    budget: float,
    *,
    run_exhaustive: bool = True,
    random_seed: int | None = 42,
    n_random_trials: int = 200,
) -> OptimisationReport:
    """Run all strategies and return a side-by-side comparison.

    Parameters
    ----------
    net:
        Base (degraded) network.
    scenario:
        Rainfall/tide scenario.
    budget:
        Repair budget.
    run_exhaustive:
        If False, skip the exhaustive search (useful for large candidate sets).
    random_seed:
        Seed for the random baseline.
    n_random_trials:
        Trials for the random baseline.

    Returns
    -------
    OptimisationReport with all plans and optimality-gap data.
    """
    base_overflow = _run(net, scenario).total_overflow_m3s
    candidates = _candidate_drains(net)

    plans: dict[str, RepairPlan] = {}

    plans["greedy"] = greedy_repair(net, scenario, budget, base_overflow=base_overflow)
    plans["severity"] = severity_baseline(net, scenario, budget, base_overflow=base_overflow)
    plans["cheapest"] = cheapest_baseline(net, scenario, budget, base_overflow=base_overflow)
    plans["ratio"] = ratio_baseline(net, scenario, budget, base_overflow=base_overflow)
    plans["random"] = random_baseline(
        net, scenario, budget,
        base_overflow=base_overflow,
        n_trials=n_random_trials,
        seed=random_seed,
    )

    exhaustive: RepairPlan | None = None
    if run_exhaustive:
        try:
            exhaustive = exhaustive_repair(net, scenario, budget, base_overflow=base_overflow)
            plans["exhaustive"] = exhaustive
        except RepairOptError:
            exhaustive = None  # too large; gap cannot be reported

    return OptimisationReport(
        scenario_id=scenario.id,
        budget=budget,
        base_overflow_m3s=base_overflow,
        candidate_drain_ids=candidates,
        plans=plans,
        exhaustive_optimal=exhaustive,
    )
