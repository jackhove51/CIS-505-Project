"""
Range-uncertainty Monte Carlo simulation for planned EV routes (tickets T9/T10).

Given a route already planned by ``dijkstra_ev``, this module *replays* the
itinerary under randomized energy consumption to estimate how often the battery
actually survives the trip.  Nothing here re-runs the search -- it only consumes
``SearchResult.itinerary``.

The search quantizes the battery into ``params.soc_levels`` discrete buckets and
rounds energy *up* (``ceil``) on every drive leg, which hides the true margin.
This simulator instead tracks the battery as a continuous value in kWh and
perturbs the per-leg consumption with lognormal noise, so the reported
feasibility reflects the real range slack rather than the quantized one.
"""
import sys
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ev_model import VehicleParams
from dijkstra_ev import SearchResult, dijkstra_ev
from load_graph import load_graph


@dataclass
class NoiseParams:
    """Lognormal noise on the per-leg consumption multiplier."""
    sigma: float = 0.15  # lognormal sigma; the multiplier's mean is held at 1.0


@dataclass
class TripResult:
    """Outcome of replaying one itinerary under a single noise realization."""
    feasible: bool
    final_soc_kwh: float
    realized_cost_dollars: float
    realized_time_h: float
    failed_leg_index: Optional[int]  # index into the itinerary; None if feasible


@dataclass
class MonteCarloResult:
    """Aggregate statistics over many replays of one route."""
    n_trials: int
    n_feasible: int
    feasibility_prob: float
    cost_percentiles: dict[str, float]   # p50/p90/p95 over FEASIBLE trials only
    time_percentiles: dict[str, float]   # p50/p90/p95 over FEASIBLE trials only
    failed_leg_counts: dict[int, int]    # leg index -> times it was first to fail


def _consumption_factor(noise: NoiseParams, rng: np.random.Generator) -> float:
    # exp(N(-s^2/2, s)) is lognormal with mean exp(-s^2/2 + s^2/2) = 1.0 exactly.
    return float(np.exp(rng.normal(-noise.sigma ** 2 / 2, noise.sigma)))


def simulate_trip(
    itinerary: list[dict],
    params: VehicleParams,
    noise: NoiseParams,
    rng: np.random.Generator,
) -> TripResult:
    """Replay one itinerary in continuous kWh.

    A fresh lognormal factor scales the consumption of every drive leg.  The trip
    fails on the first leg whose post-drive battery falls below the reserve floor;
    no emergency recharge is modelled -- we simply stop replaying that trial.
    """
    battery_kwh = params.start_soc_frac * params.battery_capacity_kwh
    reserve_kwh = params.reserve_soc_frac * params.battery_capacity_kwh

    realized_cost_dollars = 0.0
    realized_time_h       = 0.0

    for leg_index, step in enumerate(itinerary):
        if step["action"] == "drive":
            factor = _consumption_factor(noise, rng)
            used   = params.consumption_kwh_per_km * factor * step["dist_km"]
            battery_kwh     -= used
            realized_time_h += step["time_h"]
            if battery_kwh < reserve_kwh:
                return TripResult(
                    feasible=False,
                    final_soc_kwh=battery_kwh,
                    realized_cost_dollars=realized_cost_dollars,
                    realized_time_h=realized_time_h,
                    failed_leg_index=leg_index,
                )
        else:  # charge
            battery_kwh = min(
                params.battery_capacity_kwh, battery_kwh + step["energy_kwh"]
            )
            realized_cost_dollars += step["cost_dollars"]
            realized_time_h       += step["time_min"] / 60.0

    return TripResult(
        feasible=True,
        final_soc_kwh=battery_kwh,
        realized_cost_dollars=realized_cost_dollars,
        realized_time_h=realized_time_h,
        failed_leg_index=None,
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    p50, p90, p95 = np.percentile(values, [50, 90, 95])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95)}


def monte_carlo_route(
    result: SearchResult,
    params: VehicleParams,
    noise: NoiseParams,
    n_trials: int = 1000,
    seed: int = 0,
) -> MonteCarloResult:
    """Replay ``result.itinerary`` ``n_trials`` times and aggregate the outcomes.

    Cost and time percentiles are computed over FEASIBLE trials only; infeasible
    trials instead contribute to ``failed_leg_counts``.
    """
    rng = np.random.default_rng(seed)

    feasible_costs: list[float] = []
    feasible_times: list[float] = []
    failed_leg_counts: dict[int, int] = {}

    for _ in range(n_trials):
        trip = simulate_trip(result.itinerary, params, noise, rng)
        if trip.feasible:
            feasible_costs.append(trip.realized_cost_dollars)
            feasible_times.append(trip.realized_time_h)
        else:
            idx = trip.failed_leg_index
            failed_leg_counts[idx] = failed_leg_counts.get(idx, 0) + 1

    n_feasible = len(feasible_costs)
    feasibility_prob = n_feasible / n_trials if n_trials else 0.0

    return MonteCarloResult(
        n_trials=n_trials,
        n_feasible=n_feasible,
        feasibility_prob=feasibility_prob,
        cost_percentiles=_percentiles(feasible_costs),
        time_percentiles=_percentiles(feasible_times),
        failed_leg_counts=failed_leg_counts,
    )


# --------------------------------------------------------------------------- #
# Robustness layer: rank routes by how well they hold up under uncertainty.    #
# --------------------------------------------------------------------------- #

# One row of a noise sweep: (sigma, feasibility_prob, cost_p50, robustness_score).
RobustnessRow = tuple[float, float, float, float]


def robustness_score(
    mc_result: MonteCarloResult,
    params: VehicleParams,
    lam: float = 0.1,
) -> float:
    """Collapse a Monte Carlo result into one robustness number (higher is better).

        score = feasibility_prob - lam * (cost_p90 - cost_p50) / max(cost_p50, 1e-9)

    The first term rewards routes that survive range uncertainty.  The second
    subtracts a penalty proportional to the *relative* cost spread between the
    median (p50) and the 90th-percentile (p90) realized cost, so that of two
    equally-feasible routes the one with the tighter cost tail scores higher.
    ``lam`` (default 0.1) tunes how harshly that spread is penalized.

    Percentiles are taken over feasible trials only; with zero feasible trials
    there is no spread to penalize and the score is just the (zero) feasibility.
    ``params`` is accepted for interface symmetry with the rest of the module
    (and to allow cost-normalization variants) but is not needed by this formula.
    """
    feasibility = mc_result.feasibility_prob
    if mc_result.n_feasible == 0:
        return feasibility

    cost_p50 = mc_result.cost_percentiles["p50"]
    cost_p90 = mc_result.cost_percentiles["p90"]
    spread = (cost_p90 - cost_p50) / max(cost_p50, 1e-9)
    return feasibility - lam * spread


def compare_routes(
    G,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
    noise: Optional[NoiseParams] = None,
    n_trials: int = 1000,
    seed: int = 0,
    sigmas: Optional[list[float]] = None,
) -> list[RobustnessRow]:
    """Replay the SAME planned route under several noise sigmas and rank robustness.

    The route ``source -> target`` is planned once with ``dijkstra_ev``; that one
    itinerary is then Monte-Carlo replayed at each sigma in ``sigmas`` (default
    ``[0.05, 0.15, 0.30]``), reusing ``noise`` as the base config and overriding
    only its sigma.  Returns one ``(sigma, feasibility_prob, cost_p50,
    robustness_score)`` row per sigma in the given order, so a caller can watch
    robustness degrade as uncertainty grows.  Returns ``[]`` if ``target`` is
    unreachable.
    """
    if params is None:
        params = VehicleParams()
    if noise is None:
        noise = NoiseParams()
    if sigmas is None:
        sigmas = [0.05, 0.15, 0.30]

    result = dijkstra_ev(G, source, target, params)
    if not result.reached:
        return []

    rows: list[RobustnessRow] = []
    for sigma in sigmas:
        mc = monte_carlo_route(
            result, params, replace(noise, sigma=sigma), n_trials=n_trials, seed=seed
        )
        score = robustness_score(mc, params)
        rows.append((sigma, mc.feasibility_prob, mc.cost_percentiles["p50"], score))
    return rows


def print_robustness_report(
    rows: list[RobustnessRow],
    route_label: str = "",
    n_trials: Optional[int] = None,
) -> None:
    """Pretty-print a ``compare_routes`` sweep, styled like ``print_itinerary``."""
    sep  = "=" * 66
    sep2 = "-" * 66

    title = "  Robustness vs. range uncertainty"
    if route_label:
        title += f": {route_label}"

    print(f"\n{sep}")
    print(title)
    if n_trials is not None:
        print(f"  ({n_trials} trials per sigma)")
    print(sep)

    print(
        f"  {'sigma':>6s}   {'feasibility':>11s}"
        f"   {'cost p50':>9s}   {'robustness':>10s}"
    )
    print(f"  {'-' * 6}   {'-' * 11}   {'-' * 9}   {'-' * 10}")

    for sigma, feasibility, cost_p50, score in rows:
        cost_str = "n/a" if np.isnan(cost_p50) else f"${cost_p50:.2f}"
        print(
            f"  {sigma:6.2f}   {feasibility:11.2%}"
            f"   {cost_str:>9s}   {score:10.3f}"
        )

    print(f"\n{sep2}")
    if rows:
        best  = max(rows, key=lambda r: r[3])
        worst = min(rows, key=lambda r: r[3])
        print(f"  Most robust  : sigma={best[0]:.2f}   score {best[3]:.3f}   feasibility {best[1]:.2%}")
        print(f"  Least robust : sigma={worst[0]:.2f}   score {worst[3]:.3f}   feasibility {worst[1]:.2%}")
    else:
        print("  No feasible route to evaluate.")
    print(f"{sep2}\n")


def _leg_label(step: dict) -> str:
    if step["action"] == "drive":
        return f"{step['from']} -> {step['to']}"
    return f"charge @ {step['at']}"


if __name__ == "__main__":
    G = load_graph()

    SOURCE, TARGET = 0, 4
    params = VehicleParams()
    noise  = NoiseParams()

    result = dijkstra_ev(G, SOURCE, TARGET, params)
    if not result.reached:
        print("No feasible EV route found (target unreachable).", file=sys.stderr)
        sys.exit(1)

    mc = monte_carlo_route(result, params, noise, n_trials=2000, seed=0)

    src = G.nodes[SOURCE]["name"]
    dst = G.nodes[TARGET]["name"]
    sep = "=" * 66

    print(f"\n{sep}")
    print(f"  Monte Carlo range uncertainty: {src} -> {dst}")
    print(f"  trials={mc.n_trials}  sigma={noise.sigma}  (planned cost ${result.total_cost_dollars:.2f}, time {result.total_time_h:.2f} h)")
    print(sep)
    print(
        f"  Feasibility prob : {mc.feasibility_prob:7.2%}"
        f"   ({mc.n_feasible}/{mc.n_trials} trials survived)"
    )

    c = mc.cost_percentiles
    t = mc.time_percentiles
    print(f"  Cost  p50/p90/p95: ${c['p50']:7.2f} / ${c['p90']:7.2f} / ${c['p95']:7.2f}")
    print(f"  Time  p50/p90/p95: {t['p50']:7.2f} / {t['p90']:7.2f} / {t['p95']:7.2f}   h")

    print("\n  Top failing legs (first leg to drop below reserve):")
    if mc.failed_leg_counts:
        ranked = sorted(mc.failed_leg_counts.items(), key=lambda kv: kv[1], reverse=True)
        for idx, count in ranked[:5]:
            label = _leg_label(result.itinerary[idx])
            print(
                f"    leg {idx:2d}  {label:<34s}"
                f"  {count:5d} fails  ({count / mc.n_trials:6.2%})"
            )
    else:
        print(f"    none -- all {mc.n_trials} trials feasible")
    print(f"{sep}\n")

    # Robustness sweep: the same planned route under escalating range uncertainty.
    rows = compare_routes(
        G, SOURCE, TARGET, params, noise,
        n_trials=2000, seed=0, sigmas=[0.05, 0.15, 0.30],
    )
    print_robustness_report(rows, route_label=f"{src} -> {dst}", n_trials=2000)
