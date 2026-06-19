import heapq
import math
import sys
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from ev_model import VehicleParams, drive_levels, level_kwh, reserve_level, start_level
from load_graph import CHARGERS, EDGES, NODES, add_chargers, load_graph

State  = tuple[int, int]
Weight = tuple[float, float]   # (cost_dollars, time_hours) — lexicographic

_INF: Weight = (math.inf, math.inf)


@dataclass
class CostOptimalResult:
    reached: bool
    reason: str
    total_cost_dollars: float
    total_time_h: float
    route: list[int]
    itinerary: list[dict]
    states_settled: int
    heap_pushes: int


def min_cost_ev(
    G: nx.DiGraph,
    source: int,
    target: int,
    params: Optional[VehicleParams] = None,
) -> CostOptimalResult:
    if params is None:
        params = VehicleParams()

    _reserve = reserve_level(params)
    _start   = start_level(params)
    _lkwh    = level_kwh(params)

    if source == target:
        return CostOptimalResult(
            reached=True, reason="source == target",
            total_cost_dollars=0.0, total_time_h=0.0,
            route=[source], itinerary=[],
            states_settled=0, heap_pushes=0,
        )

    start_state: State = (source, _start)
    dist:   dict[State, Weight] = {start_state: (0.0, 0.0)}
    parent: dict[State, tuple]  = {}
    heap:   list[tuple[Weight, State]] = [((0.0, 0.0), start_state)]
    states_settled = 0
    heap_pushes    = 1

    while heap:
        w, state = heapq.heappop(heap)

        if w > dist.get(state, _INF):
            continue

        node, level = state
        states_settled += 1

        if node == target:
            return _reconstruct(G, state, parent, w, params, _lkwh,
                                states_settled, heap_pushes)

        for v, edge_data in G[node].items():
            d    = edge_data["distance"]
            need = drive_levels(d, params)
            new_level = level - need
            if new_level < _reserve:
                continue

            next_state: State  = (v, new_level)
            tentative:  Weight = (w[0], w[1] + d / params.avg_speed_kmh)

            if tentative < dist.get(next_state, _INF):
                dist[next_state]   = tentative
                parent[next_state] = (state, "drive", {"dist_km": d, "levels_used": need})
                heapq.heappush(heap, (tentative, next_state))
                heap_pushes += 1

        node_data = G.nodes[node]
        if node_data["is_charger"] and level < params.soc_levels:
            price   = node_data["price_per_kwh"]
            rate_kw = node_data["charge_rate_kw"]

            c_cost = _lkwh * price
            c_time = _lkwh / rate_kw
            if state not in parent or parent[state][1] == "drive":
                c_cost += params.session_fee

            next_state: State  = (node, level + 1)
            tentative:  Weight = (w[0] + c_cost, w[1] + c_time)

            if tentative < dist.get(next_state, _INF):
                dist[next_state]   = tentative
                parent[next_state] = (state, "charge", {"price": price, "rate_kw": rate_kw})
                heapq.heappush(heap, (tentative, next_state))
                heap_pushes += 1

    return CostOptimalResult(
        reached=False, reason="no feasible route (range gap)",
        total_cost_dollars=0.0, total_time_h=0.0,
        route=[], itinerary=[],
        states_settled=states_settled, heap_pushes=heap_pushes,
    )


def _reconstruct(
    G: nx.DiGraph,
    goal_state: State,
    parent: dict,
    final_weight: Weight,
    params: VehicleParams,
    _lkwh: float,
    states_settled: int,
    heap_pushes: int,
) -> CostOptimalResult:
    rev: list[State] = []
    s = goal_state
    while s in parent:
        rev.append(s)
        s = parent[s][0]
    rev.append(s)
    path = list(reversed(rev))

    route:     list[int]  = [path[0][0]]
    itinerary: list[dict] = []

    i = 1
    while i < len(path):
        _, kind, detail = parent[path[i]]

        if kind == "drive":
            prev_node, prev_lvl = path[i - 1]
            curr_node, curr_lvl = path[i]
            d        = detail["dist_km"]
            lvl_used = detail["levels_used"]
            itinerary.append({
                "action":      "drive",
                "from":        G.nodes[prev_node]["name"],
                "to":          G.nodes[curr_node]["name"],
                "dist_km":     d,
                "time_h":      d / params.avg_speed_kmh,
                "levels_used": lvl_used,
                "energy_kwh":  lvl_used * _lkwh,
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
            itinerary.append({
                "action":        "charge",
                "at":            G.nodes[charge_node]["name"],
                "levels_added":  levels_added,
                "energy_kwh":    levels_added * _lkwh,
                "cost_dollars":  cost_dollars,
                "time_h":        time_h_stop,
                "time_min":      time_h_stop * 60,
                "rate_kw":       rate_kw,
                "price_per_kwh": price,
                "soc_before":    soc_before,
                "soc_after":     soc_after,
            })

    return CostOptimalResult(
        reached=True,
        reason="",
        total_cost_dollars=final_weight[0],
        total_time_h=final_weight[1],
        route=route,
        itinerary=itinerary,
        states_settled=states_settled,
        heap_pushes=heap_pushes,
    )


def print_result(result: CostOptimalResult, G: nx.DiGraph, params: VehicleParams) -> None:
    mx   = params.soc_levels
    src  = G.nodes[result.route[0]]["name"]  if result.route else "?"
    dst  = G.nodes[result.route[-1]]["name"] if result.route else "?"
    sep  = "=" * 66
    sep2 = "-" * 66

    print(f"\n{sep}")
    print(f"  Min-Cost EV Route: {src} -> {dst}")
    print(f"{sep}")

    if not result.reached:
        print(f"  INFEASIBLE: {result.reason}")
        print(f"{sep2}\n")
        return

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
                f"  +{step['levels_added']:2d} lvl"
                f"  [SoC {step['soc_before']:2d}->{step['soc_after']:2d}/{mx}]"
                f"  ${step['cost_dollars']:6.2f}"
                f"  {step['time_min']:5.1f} min"
            )

    route_str = " -> ".join(G.nodes[n]["name"] for n in result.route)
    print(f"\n{sep2}")
    print(f"  Route    : {route_str}")
    print(f"  Cost     : ${result.total_cost_dollars:.2f}")
    print(f"  Time     : {result.total_time_h:.2f} h")
    print(f"  Settled  : {result.states_settled} states")
    print(f"  Pushes   : {result.heap_pushes}")
    print(f"{sep2}\n")


if __name__ == "__main__":
    G = load_graph(NODES, EDGES)
    add_chargers(G, CHARGERS)

    params = VehicleParams()

    for src, dst in [(0, 4), (0, 1)]:
        result = min_cost_ev(G, src, dst, params)
        print_result(result, G, params)
        if not result.reached:
            print(f"  No feasible route: {result.reason}", file=sys.stderr)
