import heapq
import sys
from dataclasses import dataclass
from typing import Optional

import networkx as nx

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

State = tuple[int, int]


@dataclass
class SearchResult:
    reached: bool
    total_weight: float
    total_cost_dollars: float
    total_time_h: float
    route: list[int]
    itinerary: list[dict]
    states_settled: int
    heap_pushes: int = 0


def dijkstra_ev(
    G: nx.DiGraph,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
    cost_upper_bound: Optional[float] = None,
) -> SearchResult:
    if params is None:
        params = VehicleParams()

    _reserve = reserve_level(params)
    _start   = start_level(params)
    _lkwh    = level_kwh(params)

    if source == target:
        return SearchResult(
            reached=True, total_weight=0.0, total_cost_dollars=0.0,
            total_time_h=0.0, route=[source], itinerary=[], states_settled=0,
        )

    start_state: State = (source, _start)

    dist: dict[State, float] = {start_state: 0.0}
    parent: dict[State, tuple] = {}
    heap: list[tuple[float, State]] = [(0.0, start_state)]
    states_settled = 0
    heap_pushes    = 1  # start_state

    while heap:
        cost, state = heapq.heappop(heap)

        if cost > dist.get(state, float("inf")):
            continue

        node, level = state
        states_settled += 1

        if node == target:
            r = _reconstruct(G, state, parent, cost, params, _lkwh, states_settled)
            r.heap_pushes = heap_pushes
            return r

        for v, edge_data in G[node].items():
            d    = edge_data["distance"]
            need = drive_levels(d, params)
            new_level = level - need
            if new_level < _reserve:
                continue

            next_state: State = (v, new_level)
            tentative = cost + drive_weight(d, params)

            if cost_upper_bound is not None and tentative > cost_upper_bound:
                continue

            if tentative < dist.get(next_state, float("inf")):
                dist[next_state] = tentative
                parent[next_state] = (
                    state, "drive", {"dist_km": d, "levels_used": need}
                )
                heapq.heappush(heap, (tentative, next_state))
                heap_pushes += 1

        node_data = G.nodes[node]
        if node_data["is_charger"] and level < params.soc_levels:
            price   = node_data["price_per_kwh"]
            rate_kw = node_data["charge_rate_kw"]

            w = charge_step_weight(price, rate_kw, params)
            if state not in parent or parent[state][1] == "drive":
                w += params.w_cost * params.session_fee

            next_state = (node, level + 1)
            tentative  = cost + w

            if cost_upper_bound is not None and tentative > cost_upper_bound:
                continue

            if tentative < dist.get(next_state, float("inf")):
                dist[next_state] = tentative
                parent[next_state] = (
                    state, "charge", {"price": price, "rate_kw": rate_kw}
                )
                heapq.heappush(heap, (tentative, next_state))
                heap_pushes += 1

    return SearchResult(
        reached=False, total_weight=float("inf"), total_cost_dollars=0.0,
        total_time_h=0.0, route=[], itinerary=[],
        states_settled=states_settled, heap_pushes=heap_pushes,
    )


def _reconstruct(
    G: nx.DiGraph,
    goal_state: State,
    parent: dict,
    total_weight: float,
    params: VehicleParams,
    _lkwh: float,
    states_settled: int,
) -> SearchResult:
    rev: list[State] = []
    s = goal_state
    while s in parent:
        rev.append(s)
        s = parent[s][0]
    rev.append(s)
    path = list(reversed(rev))

    route: list[int]      = [path[0][0]]
    itinerary: list[dict] = []
    total_cost_dollars    = 0.0
    total_time_h          = 0.0

    i = 1
    while i < len(path):
        _, kind, detail = parent[path[i]]

        if kind == "drive":
            prev_node, prev_lvl = path[i - 1]
            curr_node, curr_lvl = path[i]
            d        = detail["dist_km"]
            lvl_used = detail["levels_used"]
            t = d / params.avg_speed_kmh
            total_time_h += t
            itinerary.append({
                "action":      "drive",
                "from":        G.nodes[prev_node]["name"],
                "to":          G.nodes[curr_node]["name"],
                "dist_km":     d,
                "levels_used": lvl_used,
                "energy_kwh":  lvl_used * _lkwh,
                "time_h":      t,
                "soc_before":  prev_lvl,
                "soc_after":   curr_lvl,
            })
            route.append(curr_node)
            i += 1

        else:
            charge_node  = path[i][0]
            soc_before   = path[i - 1][1]
            price        = detail["price"]
            rate_kw      = detail["rate_kw"]
            levels_added = 0
            cost_dollars = params.session_fee
            time_h_stop  = 0.0

            while (
                i < len(path)
                and path[i][0] == charge_node
                and parent[path[i]][1] == "charge"
            ):
                levels_added += 1
                cost_dollars += _lkwh * price
                time_h_stop  += _lkwh / rate_kw
                i += 1

            soc_after = soc_before + levels_added
            total_cost_dollars += cost_dollars
            total_time_h       += time_h_stop
            itinerary.append({
                "action":        "charge",
                "at":            G.nodes[charge_node]["name"],
                "levels_added":  levels_added,
                "energy_kwh":    levels_added * _lkwh,
                "cost_dollars":  cost_dollars,
                "time_min":      time_h_stop * 60,
                "rate_kw":       rate_kw,
                "price_per_kwh": price,
                "soc_before":    soc_before,
                "soc_after":     soc_after,
            })

    return SearchResult(
        reached=True,
        total_weight=total_weight,
        total_cost_dollars=total_cost_dollars,
        total_time_h=total_time_h,
        route=route,
        itinerary=itinerary,
        states_settled=states_settled,
    )


def print_itinerary(result: SearchResult, G: nx.DiGraph, params: VehicleParams) -> None:
    mx  = params.soc_levels
    src = G.nodes[result.route[0]]["name"]  if result.route else "?"
    dst = G.nodes[result.route[-1]]["name"] if result.route else "?"

    sep  = "=" * 66
    sep2 = "-" * 66
    print(f"\n{sep}")
    print(f"  EV Route: {src} -> {dst}")
    print(f"{sep}")

    for step in result.itinerary:
        if step["action"] == "drive":
            print(
                f"  Drive  {step['from']:<18s} -> {step['to']:<18s}"
                f"  {step['dist_km']:6.1f} km"
                f"  {-step['levels_used']:+3d} lvl"
                f"  [SoC {step['soc_before']:2d}->{step['soc_after']:2d}/{mx}]"
                f"  {step['time_h'] * 60:5.1f} min"
            )
        else:
            print(
                f"  Charge {step['at']:<18s}"
                f"  ({step['rate_kw']:.0f} kW, ${step['price_per_kwh']:.2f}/kWh)"
                f"  {step['levels_added']:+3d} lvl"
                f"  [SoC {step['soc_before']:2d}->{step['soc_after']:2d}/{mx}]"
                f"  ${step['cost_dollars']:6.2f}"
                f"  {step['time_min']:5.1f} min"
            )

    route_str = " -> ".join(G.nodes[n]["name"] for n in result.route)
    print(f"\n{sep2}")
    print(f"  Route    : {route_str}")
    print(f"  Cost     : ${result.total_cost_dollars:.2f}")
    print(f"  Time     : {result.total_time_h:.2f} h")
    print(f"  Weight   : {result.total_weight:.4f}")
    print(f"  Settled  : {result.states_settled} states")
    print(f"{sep2}\n")


if __name__ == "__main__":
    G = load_graph()

    SOURCE, TARGET = 0, 4
    params = VehicleParams()
    result = dijkstra_ev(G, SOURCE, TARGET, params)

    if not result.reached:
        print("No feasible EV route found (target unreachable).", file=sys.stderr)
        sys.exit(1)

    print_itinerary(result, G, params)
