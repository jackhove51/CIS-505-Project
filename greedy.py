import math
import sys
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from dijkstra_ev import SearchResult, dijkstra_ev, print_itinerary
from ev_model import (
    VehicleParams,
    charge_step_weight,
    drive_levels,
    drive_weight,
    level_kwh,
    reserve_level,
    start_level,
)
from load_graph import load_graph


@dataclass
class GreedyParams:
    w_progress: float = 1.0
    w_price:    float = 1.0
    w_detour:   float = 0.01
    w_rate:     float = 0.001


@dataclass
class GreedyResult:
    feasible: bool
    reason: str
    route: list[int]
    charge_stops: list[dict]
    total_weight: float
    total_cost_dollars: float
    total_time_h: float
    total_distance_km: float


@dataclass
class PruningResult:
    greedy: GreedyResult
    optimal: SearchResult
    greedy_bound: float
    states_no_bound: int
    states_with_bound: int
    pushes_no_bound: int
    pushes_with_bound: int


def _path_levels(G: nx.DiGraph, u: int, v: int, params: VehicleParams) -> int:
    """
    Sum of drive_levels along the shortest-distance road path from u to v.

    Matches Dijkstra's hop-by-hop ceil accounting exactly, ensuring the
    greedy's SoC estimates are consistent with the lifted state space.
    ceil(a+b) <= ceil(a)+ceil(b), so summing per hop is more conservative
    than ceil of total distance — this is what makes the greedy's feasible
    solution genuinely feasible in Dijkstra's state space.
    """
    if u == v:
        return 0
    try:
        path = nx.shortest_path(G, u, v, weight="distance")
        return sum(
            drive_levels(G[a][b]["distance"], params)
            for a, b in zip(path, path[1:])
        )
    except nx.NetworkXNoPath:
        return math.inf


def greedy_route(
    G: nx.DiGraph,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
    gparams: Optional[GreedyParams] = None,
) -> GreedyResult:
    if params is None:
        params = VehicleParams()
    if gparams is None:
        gparams = GreedyParams()

    _reserve = reserve_level(params)
    _lkwh    = level_kwh(params)

    try:
        G_rev = G.reverse(copy=False)
        dist_to_target = dict(
            nx.single_source_dijkstra_path_length(G_rev, target, weight="distance")
        )
    except Exception:
        return GreedyResult(
            feasible=False, reason="target unreachable in graph",
            route=[], charge_stops=[], total_weight=math.inf,
            total_cost_dollars=0.0, total_time_h=0.0, total_distance_km=0.0,
        )

    if source not in dist_to_target:
        return GreedyResult(
            feasible=False, reason="no road path from source to target",
            route=[], charge_stops=[], total_weight=math.inf,
            total_cost_dollars=0.0, total_time_h=0.0, total_distance_km=0.0,
        )

    charger_nodes = [n for n, d in G.nodes(data=True) if d["is_charger"]]

    current            = source
    current_level      = start_level(params)
    route              = [source]
    charge_stops: list[dict] = []
    total_weight       = 0.0
    total_cost_dollars = 0.0
    total_time_h       = 0.0
    total_distance_km  = 0.0

    while True:
        remaining = dist_to_target.get(current, math.inf)

        if remaining == math.inf:
            return GreedyResult(
                feasible=False,
                reason=f"node {G.nodes[current]['name']} cannot reach target",
                route=route, charge_stops=charge_stops,
                total_weight=math.inf, total_cost_dollars=total_cost_dollars,
                total_time_h=total_time_h, total_distance_km=total_distance_km,
            )

        lvl_to_target = _path_levels(G, current, target, params)
        if current_level - lvl_to_target >= _reserve:
            total_weight      += drive_weight(remaining, params)
            total_time_h      += remaining / params.avg_speed_kmh
            total_distance_km += remaining
            route.append(target)
            return GreedyResult(
                feasible=True, reason="reached target",
                route=route, charge_stops=charge_stops,
                total_weight=total_weight, total_cost_dollars=total_cost_dollars,
                total_time_h=total_time_h, total_distance_km=total_distance_km,
            )

        from_current = dict(
            nx.single_source_dijkstra_path_length(G, current, weight="distance")
        )

        candidates = []
        for c in charger_nodes:
            if c == current:
                continue
            d_curr_c = from_current.get(c, math.inf)
            if d_curr_c == math.inf:
                continue
            lvl_cost = _path_levels(G, current, c, params)
            if lvl_cost == math.inf:
                continue
            arr_level = current_level - lvl_cost
            if arr_level < _reserve:
                continue
            d_c_target = dist_to_target.get(c, math.inf)
            if d_c_target >= remaining:
                continue

            lvl_c_target = _path_levels(G, c, target, params)
            need_from_c  = (lvl_c_target + _reserve
                            if lvl_c_target < math.inf else params.soc_levels)
            target_lvl   = min(need_from_c, params.soc_levels)
            charge_lvls  = max(0, target_lvl - arr_level)
            expected_kwh = charge_lvls * _lkwh
            progress     = remaining - d_c_target
            nd           = G.nodes[c]

            score = (
                gparams.w_progress * progress
                - gparams.w_price  * nd["price_per_kwh"] * expected_kwh
                - gparams.w_detour * d_curr_c
                + gparams.w_rate   * nd["charge_rate_kw"]
            )
            candidates.append((score, c, d_curr_c, arr_level))

        if not candidates:
            return GreedyResult(
                feasible=False,
                reason=(
                    f"range gap: no charger reachable from "
                    f"{G.nodes[current]['name']} that is closer to target"
                ),
                route=route, charge_stops=charge_stops,
                total_weight=math.inf, total_cost_dollars=total_cost_dollars,
                total_time_h=total_time_h, total_distance_km=total_distance_km,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best, d_to_best, arr_level = candidates[0]

        total_weight      += drive_weight(d_to_best, params)
        total_time_h      += d_to_best / params.avg_speed_kmh
        total_distance_km += d_to_best
        current_level      = arr_level
        current            = best
        route.append(current)

        lvl_to_t     = _path_levels(G, current, target, params)
        need_from_c  = (lvl_to_t + _reserve
                        if lvl_to_t < math.inf else params.soc_levels)
        target_lvl   = min(need_from_c, params.soc_levels)
        levels_added = max(0, target_lvl - current_level)

        if levels_added > 0:
            nd        = G.nodes[current]
            price     = nd["price_per_kwh"]
            rate_kw   = nd["charge_rate_kw"]
            w_stop    = (levels_added * charge_step_weight(price, rate_kw, params)
                         + params.w_cost * params.session_fee)
            cost_stop = levels_added * _lkwh * price + params.session_fee
            time_stop = levels_added * _lkwh / rate_kw

            total_weight       += w_stop
            total_cost_dollars += cost_stop
            total_time_h       += time_stop

            charge_stops.append({
                "node":          current,
                "name":          G.nodes[current]["name"],
                "kwh_added":     levels_added * _lkwh,
                "cost":          cost_stop,
                "time_h":        time_stop,
                "soc_before":    current_level,
                "soc_after":     target_lvl,
                "rate_kw":       rate_kw,
                "price_per_kwh": price,
            })
            current_level = target_lvl


def greedy_cost_upper_bound(
    G: nx.DiGraph,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
) -> float:
    return greedy_route(G, source, target, params).total_weight


def solve_with_pruning(
    G: nx.DiGraph,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
) -> PruningResult:
    if params is None:
        params = VehicleParams()

    greedy = greedy_route(G, source, target, params)
    bound  = greedy.total_weight
    ub     = bound if bound < math.inf else None

    no_bound   = dijkstra_ev(G, source, target, params, cost_upper_bound=None)
    with_bound = dijkstra_ev(G, source, target, params, cost_upper_bound=ub)

    return PruningResult(
        greedy=greedy,
        optimal=with_bound,
        greedy_bound=bound,
        states_no_bound=no_bound.states_settled,
        states_with_bound=with_bound.states_settled,
        pushes_no_bound=no_bound.heap_pushes,
        pushes_with_bound=with_bound.heap_pushes,
    )


def print_greedy_result(result: GreedyResult, G: nx.DiGraph, params: VehicleParams) -> None:
    mx   = params.soc_levels
    src  = G.nodes[result.route[0]]["name"]  if result.route else "?"
    dst  = G.nodes[result.route[-1]]["name"] if result.route else "?"
    sep  = "=" * 66
    sep2 = "-" * 66

    print(f"\n{sep}")
    print(f"  Greedy Route: {src} -> {dst}")
    print(f"{sep}")

    if not result.feasible:
        print(f"  INFEASIBLE: {result.reason}")
        print(f"{sep2}\n")
        return

    stop_idx = 0
    for i in range(len(result.route) - 1):
        u, v = result.route[i], result.route[i + 1]
        try:
            d = nx.shortest_path_length(G, u, v, weight="distance")
        except nx.NetworkXNoPath:
            d = float("nan")
        print(
            f"  Drive  {G.nodes[u]['name']:<18s} -> {G.nodes[v]['name']:<18s}"
            f"  {d:6.1f} km"
            f"  {d / params.avg_speed_kmh * 60:5.1f} min"
        )
        if stop_idx < len(result.charge_stops) and result.charge_stops[stop_idx]["node"] == v:
            s = result.charge_stops[stop_idx]
            print(
                f"  Charge {s['name']:<18s}"
                f"  ({s['rate_kw']:.0f} kW, ${s['price_per_kwh']:.2f}/kWh)"
                f"  +{s['soc_after'] - s['soc_before']:2d} lvl"
                f"  [SoC {s['soc_before']:2d}->{s['soc_after']:2d}/{mx}]"
                f"  ${s['cost']:6.2f}"
                f"  {s['time_h'] * 60:5.1f} min"
            )
            stop_idx += 1

    route_str = " -> ".join(G.nodes[n]["name"] for n in result.route)
    print(f"\n{sep2}")
    print(f"  Route    : {route_str}")
    print(f"  Cost     : ${result.total_cost_dollars:.2f}")
    print(f"  Time     : {result.total_time_h:.2f} h")
    print(f"  Weight   : {result.total_weight:.4f}")
    print(f"{sep2}\n")


def _print_t8(r: PruningResult, G: nx.DiGraph) -> None:
    sep = "=" * 66
    print(sep)
    print("  T8 Comparison")
    print(sep)
    gb  = r.greedy_bound
    ow  = r.optimal.total_weight
    print(f"  Greedy bound       : {gb:.4f}")
    print(f"  Optimal weight     : {ow:.4f}")
    gap = gb - ow
    pct = gap / ow * 100 if ow > 0 and math.isfinite(ow) else float("nan")
    print(f"  Cost gap           : {gap:.4f}  ({pct:.1f}% above optimal)")
    print(f"  States  no bound   : {r.states_no_bound}")
    print(f"  States  pruned     : {r.states_with_bound}  (saved {r.states_no_bound - r.states_with_bound})")
    saved_p = r.pushes_no_bound - r.pushes_with_bound
    pct_p   = saved_p / r.pushes_no_bound * 100 if r.pushes_no_bound > 0 else 0.0
    print(f"  Pushes  no bound   : {r.pushes_no_bound}")
    print(f"  Pushes  pruned     : {r.pushes_with_bound}  (saved {saved_p},  {pct_p:.1f}%)")
    print(sep + "\n")


if __name__ == "__main__":
    G = load_graph()
    params = VehicleParams()

    print("\n### Chicago (0) -> Cleveland (4) ###")
    r1 = solve_with_pruning(G, 0, 4, params)
    print_greedy_result(r1.greedy, G, params)
    print_itinerary(r1.optimal, G, params)
    _print_t8(r1, G)

    print("### Chicago (0) -> Columbus (1) -- greedy suboptimal ###")
    r2 = solve_with_pruning(G, 0, 1, params)
    print_greedy_result(r2.greedy, G, params)
    print_itinerary(r2.optimal, G, params)
    _print_t8(r2, G)
