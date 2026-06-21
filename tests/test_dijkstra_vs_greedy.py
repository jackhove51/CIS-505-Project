"""
The lone test in this file proves that the greedy algorithm is not always optimal compared to Dijkstra's algorithm.
"""

import networkx as nx
import pytest

from dijkstra_ev import dijkstra_ev
from ev_model import VehicleParams
from greedy import greedy_route, GreedyParams


def test_dijkstra_vs_greedy_with_greedy_failure():
    """
    This test demonstrates that the greedy algorithm can produce a suboptimal solution compared to Dijkstra's algorithm. The distances are carefully manipulated to create a scenario where the greedy algorithm chooses a more expensive route due to its local decision-making, while Dijkstra's algorithm finds the optimal path with lower overall cost.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.40, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=350.0)
    graph.add_edge(1, 2, distance=125.0)
    graph.add_edge(2, 3, distance=325.0)

    params = VehicleParams(
        battery_capacity_kwh=100.0
    )

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    dijkstra_result = dijkstra_ev(graph, 0, 3, params)
    greedy_result = greedy_route(graph, 0, 3, params, gparams)

    assert dijkstra_result.reached
    assert greedy_result.feasible

    assert dijkstra_result.route == [0, 1, 2, 3]
    assert dijkstra_result.total_cost_dollars == pytest.approx(28.5)
    assert dijkstra_result.total_time_h == pytest.approx(9.4)

    assert greedy_result.route == [0, 1, 3]
    assert len(greedy_result.charge_stops) == 1
    assert greedy_result.charge_stops[0]["node"] == 1
    assert greedy_result.charge_stops[0]["soc_before"] == 6
    assert greedy_result.charge_stops[0]["soc_after"] == 20
    assert greedy_result.total_cost_dollars == pytest.approx(35.0)

    assert greedy_result.total_time_h == pytest.approx(dijkstra_result.total_time_h)

    assert greedy_result.route != dijkstra_result.route
    assert greedy_result.total_cost_dollars > dijkstra_result.total_cost_dollars
    assert greedy_result.total_cost_dollars - dijkstra_result.total_cost_dollars == pytest.approx(6.5)