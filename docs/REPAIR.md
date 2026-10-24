# Phase 4: Repair Optimisation

## Problem statement

Given the drainage network in its current *degraded* state and a repair budget B,
choose a subset of degraded drains to restore to their design capacity so as to
maximise total overflow reduction.

Formally: let C be the set of drains with `capacity_factor < 1`.  A *repair
plan* R ⊆ C is feasible if Σ repair_cost(d) ≤ B.  We want:

    argmax_{R ⊆ C, cost(R) ≤ B}  [ overflow(base) − overflow(network repaired with R) ]

---

## Interaction effects and why they matter

Drain repairs interact via the flow network.

**Sub-additive gains** (joint < sum of marginals): when two drains both feed
the same downstream bottleneck, repairing the second one beyond what the
bottleneck can absorb gives diminishing return.

**Super-additive gains** (joint > sum of marginals): when two drains form a
series path that is fully blocked, repairing either one alone yields zero gain,
but repairing both together opens the path completely.

All three strategies (greedy, exhaustive, and baselines) use the same
max-flow engine to evaluate any repair plan, so interactions are always
accounted for in the *evaluation* of a plan.  The difference is whether the
*selection* accounts for them.

---

## Strategies

### Greedy (marginal-gain, re-evaluating)

At each step, evaluate the marginal overflow reduction of each remaining
candidate in the *current repaired state* (not the base state).  Pick the
candidate with the highest gain that fits in the remaining budget.  Repeat.

This is the greedy algorithm for submodular maximisation.  It carries a
(1 − 1/e) ≈ 0.63 approximation guarantee when the objective is monotone
submodular, and in practice often finds the optimum.

However, it can be suboptimal.  The classic counter-example (reproduced in
`tests/test_repair.py::test_greedy_is_suboptimal_on_density_example`):

| Drain | Individual gain | Cost | Gain/cost |
|-------|----------------|------|-----------|
| D_h   | 2.4            | 10   | 0.24      |
| D_m1  | 1.5            | 5    | 0.30      |
| D_m2  | 1.5            | 5    | 0.30      |

Budget = 10.  Greedy picks D_h first (highest absolute gain = 2.4), exhausting
the budget.  But D_m1 + D_m2 costs 10 and gives 3.0 > 2.4.  Greedy is
suboptimal.

### Exhaustive (exact optimum)

Enumerate all 2^n subsets of candidates, evaluate each feasible one, and return
the best.  This is O(2^n) max-flow runs.  A hard ceiling of 2^18 = 262 144
subsets is enforced to prevent runaway runtimes; above that, `RepairOptError`
is raised and the caller should use greedy instead.

For the bundled 14-drain simulated network (n ≤ 4 degraded drains = 16
subsets), exhaustive runs in milliseconds.

### Baselines

| Baseline | Ranking criterion | Notes |
|----------|------------------|-------|
| **severity** | decreasing marginal overflow gain at base state | Like greedy step 1, no re-evaluation |
| **cheapest** | increasing repair cost | Maximises drain count, not value |
| **ratio** | decreasing gain-per-cost-unit at base state | Fractional knapsack, applied integrally |
| **random** | random feasible subsets, best of n_trials | Seed-reproducible; default 200 trials |

All baselines rank candidates once from the base state, then fill greedily
in that fixed order (no re-evaluation after each choice).

---

## Optimality-gap report

`compare_strategies` runs all five strategies and computes each one's
*optimality gap* relative to the exhaustive optimum:

    gap(strategy) = (opt_reduction − strategy_reduction) / opt_reduction

A gap of 0 means the strategy found the optimum.  When exhaustive was not run
(e.g., too many candidates), the gap is reported as `?`.

---

## CLI usage

```bash
# Default: budget=50, design scenario, all strategies including exhaustive
floodcut repair

# Custom budget and scenario
floodcut repair --budget 80 --scenario extreme

# Skip exhaustive search (fast, no gap reporting)
floodcut repair --budget 80 --no-exhaustive

# Custom network file
floodcut repair --network path/to/net.json --budget 60
```

The output table shows cost, overflow reduction, residual overflow after
repair, optimality gap, and the drains chosen by each strategy.  Below the
table, a per-drain breakdown shows base-state marginal gain and gain-per-cost
for each candidate, useful for manual prioritisation.

---

## Implementation notes

* `_apply_repairs` returns a new `DrainageNetwork` sharing all immutable
  objects with the original; it only replaces the `drains` dict entry for
  repaired drains.  The original network is never mutated.

* `cross_check=False` is used throughout Phase 4 (as in Phase 3) because many
  max-flow calls are made per optimisation run.  The cross-check against
  NetworkX was already validated exhaustively in Phase 2.

* Drain costs are in abstract *cost units* matching `Drain.repair_cost`.  No
  real cost data is available for the simulated network; values were chosen
  to make the optimisation interesting (non-trivial greedy counter-examples
  exist at realistic budget levels).
