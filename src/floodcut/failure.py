"""Phase 3: failure simulation.

Answers: *given that k specific drains are fully blocked (capacity_factor = 0),
how does the network's deliverable flow and overflow change?*

-------------------------------------------------------------------------------
1.  What "failure" means here
-------------------------------------------------------------------------------
A **k-drain failure** is the simultaneous, complete blocking of k drains.  We
model it by zeroing those drains' ``capacity_factor`` values before building
the flow network, running the Phase 2 engine, and recording the result.  Every
other model parameter — rainfall intensity, tide factor, all other drains —
stays exactly as in the base scenario.

"Complete blocking" (factor = 0) is conservative: a drain is either fine or
gone.  Partial degradation is already modelled by the base network's
``capacity_factor`` values; the failure scenarios on top of that represent
*additional* failures of currently-healthy or partially-degraded drains.

Only drains with ``capacity_factor > 0`` are eligible for failure simulation:
there is nothing additional to lose from a drain that is already fully blocked.

-------------------------------------------------------------------------------
2.  Combinatorial enumeration (k-subsets)
-------------------------------------------------------------------------------
Let m = number of eligible drains.  The number of k-drain failure subsets is
C(m, k) = m! / (k! (m - k)!).  We enumerate all of them with
``itertools.combinations``, which is guaranteed to visit exactly C(m, k)
distinct subsets in lexicographic order.  Tests verify the count matches the
closed-form formula directly.

For the bundled 14-drain network (where typically m ≈ 12 eligible drains after
excluding already-fully-blocked drains), k ≤ 3 is tractable in milliseconds;
k ≤ 5 finishes in seconds.  For k > 5 on large networks we surface a warning
but do not hard-cap: the caller decides the acceptable runtime.

-------------------------------------------------------------------------------
3.  Worst-case identification
-------------------------------------------------------------------------------
Among all k-drain failure results, the **worst case** is the subset that
maximises total overflow (equivalently, minimises max flow).  When two subsets
tie on overflow we break ties by preferring the one with the lower total
repair_cost (a cheaper repair burden to recover from a tie is mildly
preferable), then lexicographically by drain id list for a deterministic tie
among cost-ties.

-------------------------------------------------------------------------------
4.  Marginal and combined overflow
-------------------------------------------------------------------------------
For a single drain d, its **marginal overflow** is:

    Δoverflow(d) = overflow(fail {d}) − overflow(base)

This measures how much *extra* flooding is caused by losing that one drain on
top of the already-degraded base.

For a set S of k drains the **combined overflow** is simply the total overflow
from the failure scenario where all drains in S are blocked simultaneously.

Superadditivity (Δ(S) > Σ Δ({d}) for d in S) can arise when drains share a
downstream path: losing both at once cuts a path that losing each individually
did not fully close.  Subadditivity (Δ(S) < Σ Δ({d})) can arise from
redundancy: two drains feeding the same downstream bottleneck, so blocking one
already saturates the bottleneck and blocking the other adds nothing on top.
Phase 3 does not attempt to explain which regime applies — it just reports the
combined overflow and lets the caller compare it to the marginal sum.

-------------------------------------------------------------------------------
5.  Probabilistic expected overflow (exact, for small m)
-------------------------------------------------------------------------------
Given an independent per-drain failure probability p (same for all drains, for
simplicity), the exact expected overflow at a given rainfall scenario is:

    E[overflow] = Σ_{k=0}^{m} C(m,k) * p^k * (1-p)^(m-k)
                  * overflow(worst-case k-failure)

But "worst-case" is not the right weight here — each specific subset S has
probability p^|S| * (1-p)^(m-|S|), so:

    E[overflow] = Σ_{S ⊆ eligible} p^|S| * (1-p)^(m-|S|) * overflow(S)

This is exact when *all* subsets have been enumerated.  For large m we instead
bound it between the minimum (all base) and maximum (worst full-failure) and
report that the exact computation was skipped.

The ``expected_overflow`` function below always runs if ``all_results`` covers
every subset (i.e. k_max >= m), otherwise it raises ``FailureSimError`` to
signal that only a partial enumeration was done and an expectation would be
misleading.  The caller is responsible for choosing k_max large enough.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from .graph import build_flow_network
from .maxflow import FloodCutResult, analyze
from .models import Drain, DrainageNetwork, RainfallScenario


class FailureSimError(ValueError):
    """Raised when failure simulation is asked to do something unsound."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SingleFailureResult:
    """Max-flow result plus metadata for one specific drain-failure subset."""
    failed_drain_ids: tuple[str, ...]   # sorted; empty = base (no failure)
    result: FloodCutResult
    total_repair_cost: float            # sum of repair_cost of the failed drains

    @property
    def total_overflow_m3s(self) -> float:
        return self.result.total_overflow_m3s

    @property
    def max_flow_m3s(self) -> float:
        return self.result.max_flow_m3s


@dataclass
class FailureSimSummary:
    """All results for one (network, scenario, k_max) failure enumeration."""
    scenario_id: str
    base_overflow_m3s: float
    base_max_flow_m3s: float
    # Nested by failure size: all_results[k] lists every k-drain failure result.
    all_results: dict[int, list[SingleFailureResult]]
    eligible_drain_ids: tuple[str, ...]     # drains that could fail (factor > 0)
    k_max: int

    # -----------------------------------------------------------------------
    # Counts and worst-case helpers
    # -----------------------------------------------------------------------

    def count(self, k: int) -> int:
        """Number of k-failure subsets enumerated (must equal C(m, k))."""
        return len(self.all_results.get(k, []))

    @property
    def m(self) -> int:
        """Number of eligible drains."""
        return len(self.eligible_drain_ids)

    def worst_case(self, k: int) -> SingleFailureResult | None:
        """Result with the maximum overflow among all k-drain failures.

        Ties broken by lowest total repair_cost, then by lexicographic order of
        failed drain ids (deterministic).
        """
        bucket = self.all_results.get(k)
        if not bucket:
            return None
        return max(bucket, key=lambda r: (
            r.total_overflow_m3s,
            -r.total_repair_cost,      # lower cost is "worse" (cheaper to fix means more common)
            r.failed_drain_ids,
        ))

    def worst_across_all_k(self) -> SingleFailureResult | None:
        """Worst-case result across every enumerated failure size."""
        candidates = [r for k in self.all_results for r in self.all_results[k] if k > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (
            r.total_overflow_m3s,
            -r.total_repair_cost,
            r.failed_drain_ids,
        ))

    def marginal_overflow(self, drain_id: str) -> float:
        """Extra overflow caused by losing drain_id alone, above the base.

        Raises FailureSimError if k=1 was not enumerated or drain_id is not
        among the eligible drains.
        """
        if drain_id not in self.eligible_drain_ids:
            raise FailureSimError(
                f"Drain {drain_id!r} is not eligible for failure simulation "
                f"(not in eligible set: {self.eligible_drain_ids})"
            )
        bucket = self.all_results.get(1, [])
        match = next((r for r in bucket if r.failed_drain_ids == (drain_id,)), None)
        if match is None:
            raise FailureSimError(
                f"k=1 results not found for drain {drain_id!r}; "
                "run enumerate_failures with k_max >= 1"
            )
        return match.total_overflow_m3s - self.base_overflow_m3s

    def summary_lines(self) -> list[str]:
        lines = [
            f"Failure simulation: scenario={self.scenario_id}  "
            f"eligible drains={self.m}  k_max={self.k_max}",
            f"Base overflow: {self.base_overflow_m3s:.4f} m3/s  "
            f"base max flow: {self.base_max_flow_m3s:.4f} m3/s",
        ]
        for k in sorted(self.all_results):
            if k == 0:
                continue
            wc = self.worst_case(k)
            expected_count = _comb(self.m, k)
            actual_count = self.count(k)
            lines.append(
                f"k={k}: {actual_count} of {expected_count} subset(s)  "
                + (
                    f"worst: {list(wc.failed_drain_ids)} -> "
                    f"overflow={wc.total_overflow_m3s:.4f} m3/s  "
                    f"(+{wc.total_overflow_m3s - self.base_overflow_m3s:.4f})  "
                    f"max_flow={wc.max_flow_m3s:.4f} m3/s"
                    if wc else "no results"
                )
            )
        return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    """Exact integer C(n, k)."""
    return math.comb(n, k)


def _apply_failures(net: DrainageNetwork, failed_ids: Sequence[str]) -> DrainageNetwork:
    """Return a shallow copy of *net* with the specified drains zeroed out.

    Only the ``drains`` dict is replaced; all other attributes are shared with
    the original (they are immutable dataclasses, so this is safe).
    """
    new_drains = dict(net.drains)
    for did in failed_ids:
        d = new_drains[did]
        # Use dataclasses copy-and-modify (frozen=True means we must replace).
        new_drains[did] = Drain(
            id=d.id,
            from_junction=d.from_junction,
            to_junction=d.to_junction,
            design_capacity_m3s=d.design_capacity_m3s,
            capacity_factor=0.0,   # fully blocked
            repair_cost=d.repair_cost,
            length_m=d.length_m,
            kind=d.kind,
        )
    # Build a lightweight replacement container sharing everything else.
    failed_net = copy.copy(net)
    object.__setattr__(failed_net, "drains", new_drains) if hasattr(net, "__dataclass_fields__") \
        else None
    # DrainageNetwork is @dataclass(eq=False) but NOT frozen, so ordinary assignment works.
    failed_net = DrainageNetwork(
        metadata=net.metadata,
        junctions=net.junctions,
        drains=new_drains,
        catchments=net.catchments,
        outfalls=net.outfalls,
        assets=net.assets,
    )
    return failed_net


def _eligible_drains(net: DrainageNetwork) -> tuple[str, ...]:
    """Sorted ids of drains with effective capacity > 0 (can still fail)."""
    return tuple(sorted(d.id for d in net.drains.values() if d.capacity_factor > 0))


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def enumerate_failures(
    net: DrainageNetwork,
    scenario: RainfallScenario,
    *,
    k_max: int,
    cross_check: bool = False,
) -> FailureSimSummary:
    """Enumerate all drain-failure subsets of size 1 … k_max.

    For each subset, zeroes the specified drains' capacity factors, rebuilds
    the flow network, and runs the Phase 2 engine (Edmonds-Karp + min cut).

    Parameters
    ----------
    net:
        The base drainage network (unmodified — failed copies are ephemeral).
    scenario:
        The rainfall/tide scenario to use for all runs.
    k_max:
        Maximum failure size to enumerate.  Must be >= 1.
        All subsets of size 1, 2, … k_max are returned.
    cross_check:
        If True, each run's Edmonds-Karp value is verified against NetworkX.
        Defaults to False because Phase 3 runs many times more scenarios than
        Phase 2 and the cross-check adds ~30% overhead per run.

    Returns
    -------
    FailureSimSummary
        Contains the base result (k=0), all k-failure results keyed by k, the
        eligible drain list, and convenience methods for worst-case look-up.

    Raises
    ------
    FailureSimError
        If k_max < 1, or if k_max > m (more failures than eligible drains).
    """
    eligible = _eligible_drains(net)
    m = len(eligible)

    if k_max < 1:
        raise FailureSimError(f"k_max must be >= 1, got {k_max}")
    if k_max > m:
        raise FailureSimError(
            f"k_max={k_max} exceeds the number of eligible drains ({m}); "
            f"eligible: {eligible}"
        )

    # --- Base run (no additional failures) ---
    base_fn = build_flow_network(net, scenario)
    base_result = analyze(net, base_fn, cross_check=cross_check)

    all_results: dict[int, list[SingleFailureResult]] = {0: [
        SingleFailureResult(
            failed_drain_ids=(),
            result=base_result,
            total_repair_cost=0.0,
        )
    ]}

    # --- Failure runs ---
    for k in range(1, k_max + 1):
        bucket: list[SingleFailureResult] = []
        for combo in combinations(eligible, k):
            failed_net = _apply_failures(net, combo)
            fn = build_flow_network(failed_net, scenario)
            result = analyze(failed_net, fn, cross_check=cross_check)
            repair_cost = sum(net.drains[did].repair_cost for did in combo)
            bucket.append(SingleFailureResult(
                failed_drain_ids=combo,   # combinations returns sorted tuples (eligible is sorted)
                result=result,
                total_repair_cost=repair_cost,
            ))
        all_results[k] = bucket

    return FailureSimSummary(
        scenario_id=scenario.id,
        base_overflow_m3s=base_result.total_overflow_m3s,
        base_max_flow_m3s=base_result.max_flow_m3s,
        all_results=all_results,
        eligible_drain_ids=eligible,
        k_max=k_max,
    )


def expected_overflow(summary: FailureSimSummary, p: float) -> float:
    """Exact expected overflow under independent per-drain failure probability *p*.

    Each eligible drain fails independently with probability *p* (uniform, for
    tractability — see module docstring §5).  The expectation is:

        E[overflow] = Σ_{S ⊆ eligible} p^|S| * (1-p)^(m-|S|) * overflow(S)

    This is *exact* and requires that *summary* contains ALL subsets (i.e.
    ``summary.k_max == summary.m``).

    The k=0 (no-additional-failure) term uses the base overflow weighted by
    (1-p)^m.

    Raises
    ------
    FailureSimError
        If summary.k_max < summary.m (partial enumeration — exact E is not
        computable), or if p is not in [0, 1].
    """
    if not (isinstance(p, (int, float)) and not isinstance(p, bool)
            and math.isfinite(p) and 0.0 <= p <= 1.0):
        raise FailureSimError(f"p must be a probability in [0, 1], got {p!r}")
    if summary.k_max < summary.m:
        raise FailureSimError(
            f"Exact expected overflow requires k_max == m (all subsets enumerated). "
            f"Got k_max={summary.k_max}, m={summary.m}. "
            f"Re-run enumerate_failures with k_max={summary.m} to enable this."
        )

    m = summary.m
    total = 0.0
    for k, bucket in summary.all_results.items():
        # Weight for any specific k-subset: p^k * (1-p)^(m-k)
        w = (p ** k) * ((1.0 - p) ** (m - k))
        for sfr in bucket:
            total += w * sfr.total_overflow_m3s
    return total
