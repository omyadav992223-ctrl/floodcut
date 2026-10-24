"""Command-line interface (Phase 1–4: info, validate, demand, maxflow, failure, repair)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

from . import __version__
from .failure import FailureSimError, enumerate_failures, expected_overflow
from .graph import build_flow_network, build_physical_graph
from .io import DataError, load_network, load_scenarios
from .maxflow import FloodCutError, analyze
from .models import DrainageNetwork, RainfallScenario
from .repair import RepairOptError, compare_strategies, _candidate_drains
from .validation import errors, validate_network

BANNER = "*** SIMULATED DATA: synthetic network, NOT real Mumbai infrastructure ***"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "simulated"


def _load(args) -> DrainageNetwork:
    path = Path(args.network) if args.network else default_data_dir() / "mumbai_sim_network.json"
    net = load_network(path)
    print(BANNER if net.metadata.is_simulated else "Dataset flagged as NOT simulated")
    print(f"Dataset: {net.metadata.name}")
    return net


def cmd_validate(args) -> int:
    net = _load(args)
    issues = validate_network(net)
    for i in issues:
        print(i)
    n_err = len(errors(issues))
    print(f"{n_err} error(s), {len(issues) - n_err} warning(s)")
    return 1 if n_err else 0


def cmd_info(args) -> int:
    net = _load(args)
    print(f"Source:  {net.metadata.source}")
    for k, v in net.counts().items():
        print(f"  {k:16s} {v}")
    g = build_physical_graph(net)
    simple = nx.DiGraph(g)
    parallel = {k: n for k, n in Counter((u, v) for u, v, _ in g.edges(keys=True)).items() if n > 1}
    print("Physical graph (MultiDiGraph):")
    print(f"  nodes={g.number_of_nodes()}  edges={g.number_of_edges()}")
    print(f"  acyclic (DAG): {nx.is_directed_acyclic_graph(simple)}")
    print(f"  parallel drain groups: {parallel if parallel else 'none'}")
    blocked = [d.id for d in net.drains.values() if d.capacity_factor < 1.0]
    print(f"  degraded drains (capacity_factor < 1): {blocked if blocked else 'none'}")
    issues = validate_network(net)
    for i in issues:
        print(i)
    return 1 if errors(issues) else 0


def _get_scenario(scenarios: dict[str, RainfallScenario], name: str) -> RainfallScenario | None:
    if name in scenarios:
        return scenarios[name]
    if name == "moderate" and "design" in scenarios:
        return RainfallScenario(
            id="moderate",
            name="Moderate rain (design storm)",
            intensity_mm_h=scenarios["design"].intensity_mm_h,
            tide_factor=scenarios["design"].tide_factor,
            description="Alias for design storm",
        )
    return None


def cmd_demand(args) -> int:
    net = _load(args)
    scen_path = Path(args.scenarios) if args.scenarios else default_data_dir() / "rainfall_scenarios.json"
    scenarios = load_scenarios(scen_path)
    sc = _get_scenario(scenarios, args.scenario)
    if sc is None:
        print(f"Unknown scenario {args.scenario!r}. Available: {', '.join(scenarios)}", file=sys.stderr)
        return 2
    fn = build_flow_network(net, sc)
    print(f"Scenario: {sc.id} ({sc.name}): {sc.intensity_mm_h} mm/h, tide factor {sc.tide_factor}")
    print(f"{'catchment':10s} {'junction':9s} {'area_ha':>8s} {'C':>5s} {'demand m3/s':>12s}")
    for c in net.catchments.values():
        print(f"{c.id:10s} {c.junction_id:9s} {c.area_ha:8.1f} {c.runoff_coefficient:5.2f} "
              f"{c.runoff_m3s(sc.intensity_mm_h):12.3f}")
    out_cap = sum(o.effective_capacity_m3s(sc.tide_factor) for o in net.outfalls.values())
    print(f"Total rainfall demand:        {fn.total_demand_m3s:8.3f} m3/s")
    print(f"Total outfall capacity:       {out_cap:8.3f} m3/s (tide-adjusted)")
    print("NOTE: this is only a coarse input summary. Deliverable flow and overflow are "
          "computed by max-flow in Phase 2.")
    return 0


def cmd_maxflow(args) -> int:
    net = _load(args)
    scen_path = Path(args.scenarios) if args.scenarios else default_data_dir() / "rainfall_scenarios.json"
    scenarios = load_scenarios(scen_path)
    sc = _get_scenario(scenarios, args.scenario)
    if sc is None:
        print(f"Unknown scenario {args.scenario!r}. Available: {', '.join(scenarios)}", file=sys.stderr)
        return 2
    fn = build_flow_network(net, sc)
    result = analyze(net, fn, cross_check=not args.no_cross_check)
    for line in result.summary_lines():
        print(line)
    print(result.note_on_overflow_uniqueness)
    return 0


def cmd_failure(args) -> int:
    net = _load(args)
    scen_path = Path(args.scenarios) if args.scenarios else default_data_dir() / "rainfall_scenarios.json"
    scenarios = load_scenarios(scen_path)
    sc = _get_scenario(scenarios, args.scenario)
    if sc is None:
        print(f"Unknown scenario {args.scenario!r}. Available: {', '.join(scenarios)}", file=sys.stderr)
        return 2

    k_max = args.k
    eligible_count = sum(1 for d in net.drains.values() if d.capacity_factor > 0)
    if k_max > eligible_count:
        print(f"k={k_max} exceeds eligible drain count ({eligible_count}); "
              f"clamping to {eligible_count}.", file=sys.stderr)
        k_max = eligible_count
    if k_max < 1:
        print("k must be >= 1", file=sys.stderr)
        return 2

    try:
        summary = enumerate_failures(net, sc, k_max=k_max)
    except FailureSimError as exc:
        print(f"Failure simulation error: {exc}", file=sys.stderr)
        return 2

    for line in summary.summary_lines():
        print(line)

    # Per-drain marginal table (requires k=1 enumeration, which k_max >= 1 guarantees).
    print("\nPer-drain marginal overflow (k=1):")
    print(f"  {'drain':8s}  {'marginal m3/s':>14s}  {'base cap m3/s':>14s}  {'repair cost':>11s}")
    for did in summary.eligible_drain_ids:
        d = net.drains[did]
        marginal = summary.marginal_overflow(did)
        print(f"  {did:8s}  {marginal:14.4f}  {d.effective_capacity_m3s:14.4f}  {d.repair_cost:11.2f}")

    # Worst-case per k.
    print("\nWorst-case failure subsets:")
    for k in range(1, k_max + 1):
        wc = summary.worst_case(k)
        if wc:
            extra = wc.total_overflow_m3s - summary.base_overflow_m3s
            print(f"  k={k}: {list(wc.failed_drain_ids)}  overflow={wc.total_overflow_m3s:.4f} m3/s "
                  f"(+{extra:.4f})  max_flow={wc.max_flow_m3s:.4f} m3/s  "
                  f"repair_cost={wc.total_repair_cost:.2f}")

    # Expected overflow if full enumeration was requested (k_max == m).
    if k_max == summary.m and args.prob is not None:
        try:
            p = float(args.prob)
            e_ov = expected_overflow(summary, p)
            print(f"\nExpected overflow (p={p} per drain): {e_ov:.4f} m3/s")
        except (FailureSimError, ValueError) as exc:
            print(f"\nCould not compute expected overflow: {exc}", file=sys.stderr)

    return 0


def cmd_repair(args) -> int:
    net = _load(args)
    scen_path = Path(args.scenarios) if args.scenarios else default_data_dir() / "rainfall_scenarios.json"
    scenarios = load_scenarios(scen_path)
    sc = _get_scenario(scenarios, args.scenario)
    if sc is None:
        print(f"Unknown scenario {args.scenario!r}. Available: {', '.join(scenarios)}", file=sys.stderr)
        return 2
    budget = args.budget

    candidates = _candidate_drains(net)
    if not candidates:
        print("No degraded drains found — nothing to repair.")
        return 0

    print(f"\nScenario: {sc.id} ({sc.name})  budget: {budget:.2f}")
    print(f"Repair candidates ({len(candidates)}): {list(candidates)}")
    print()

    run_exhaustive = not args.no_exhaustive
    try:
        report = compare_strategies(
            net, sc, budget,
            run_exhaustive=run_exhaustive,
            random_seed=args.seed,
            n_random_trials=args.random_trials,
        )
    except RepairOptError as exc:
        print(f"Repair optimisation error: {exc}", file=sys.stderr)
        return 2

    for line in report.summary_lines():
        print(line)

    # Per-candidate repair value table (base-state marginal gain).
    print("\nPer-drain repair value (base-state marginal gain):")
    print(f"  {'drain':8s}  {'cost':>6s}  {'gain m3/s':>10s}  {'gain/cost':>10s}  factor")
    from .repair import _run
    base_ov = report.base_overflow_m3s
    for did in candidates:
        d = net.drains[did]
        repaired_ov = _run(net, sc, [did]).total_overflow_m3s
        gain = max(0.0, base_ov - repaired_ov)
        ratio = gain / d.repair_cost if d.repair_cost > 1e-9 else float("inf")
        print(f"  {did:8s}  {d.repair_cost:6.2f}  {gain:10.4f}  {ratio:10.4f}  {d.capacity_factor:.2f}")

    return 0


def cmd_app(args) -> int:
    import subprocess
    import sys
    from pathlib import Path

    app_file = Path(__file__).resolve().parent / "app.py"
    port = str(args.port)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.port", port]
    if getattr(args, "headless", False):
        cmd.extend(["--server.headless", "true"])
    print(f"Launching FloodCut Streamlit dashboard on port {port}...")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="floodcut",
                                description="FloodCut (Phase 1–6: frontend and visualization)")
    p.add_argument("--version", action="version", version=f"floodcut {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (
        ("info",      cmd_info,      "summarise the network and its graph"),
        ("validate",  cmd_validate,  "validate the network data"),
        ("demand",    cmd_demand,    "show rainfall demand for a scenario"),
        ("maxflow",   cmd_maxflow,   "run the max-flow engine: deliverable flow, min cut, overflow"),
        ("failure",   cmd_failure,   "enumerate k-drain failures; worst-case and marginal overflow"),
        ("repair",    cmd_repair,    "budget-based drain repair optimisation; compare strategies"),
        ("app",       cmd_app,       "launch interactive Streamlit dashboard"),
        ("dashboard", cmd_app,       "alias for app"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--network", help="path to a network JSON (default: bundled simulated data)")
        if name in ("app", "dashboard"):
            sp.add_argument("--port", type=int, default=8501, help="server port (default: 8501)")
            sp.add_argument("--headless", action="store_true", help="run in headless mode")
        if name in ("demand", "maxflow", "failure", "repair"):
            sp.add_argument("--scenarios", help="path to a scenarios JSON")
            sp.add_argument("--scenario", default="design", help="scenario id (default: design)")
        if name == "maxflow":
            sp.add_argument("--no-cross-check", action="store_true",
                            help="skip the NetworkX cross-check (faster on large graphs)")
        if name == "failure":
            sp.add_argument("-k", type=int, default=2,
                            help="maximum failure size to enumerate (default: 2)")
            sp.add_argument("--prob", type=float, default=None,
                            help="per-drain failure probability for expected overflow "
                                 "(only computed when k == number of eligible drains)")
        if name == "repair":
            sp.add_argument("--budget", type=float, default=50.0,
                            help="repair budget in cost units (default: 50)")
            sp.add_argument("--no-exhaustive", action="store_true",
                            help="skip exhaustive search (useful for large candidate sets)")
            sp.add_argument("--random-trials", type=int, default=200,
                            help="number of random trials for the random baseline (default: 200)")
            sp.add_argument("--seed", type=int, default=42,
                            help="random seed for reproducibility (default: 42)")
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 2
    except FloodCutError as exc:
        print(f"Max-flow engine error: {exc}", file=sys.stderr)
        return 2
    except FailureSimError as exc:
        print(f"Failure simulation error: {exc}", file=sys.stderr)
        return 2
    except RepairOptError as exc:
        print(f"Repair optimisation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
