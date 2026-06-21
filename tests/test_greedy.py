import networkx as nx
from math import inf
import pytest

from ev_model import VehicleParams
from greedy import greedy_route, GreedyParams, greedy_cost_upper_bound


@pytest.fixture
def params():
    return VehicleParams(
        battery_capacity_kwh=100.0
    )


def test_greedy_selection_with_early_exit(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=100.0)
    graph.add_edge(0, 2, distance=100.0)
    graph.add_edge(1, 3, distance=100.0)
    graph.add_edge(2, 3, distance=100.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 3, params, gparams)
    assert result.feasible
    assert result.route == [0, 3]
    assert result.total_cost_dollars == pytest.approx(0.0)


def test_greedy_selection_isolating_price(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.30, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=300.0)
    graph.add_edge(0, 2, distance=300.0)
    graph.add_edge(1, 3, distance=300.0)
    graph.add_edge(2, 3, distance=300.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 3, params, gparams)
    assert result.feasible
    assert result.route == [0, 1, 3]
    assert result.charge_stops[0]["cost"] == pytest.approx(9.0)


def test_greedy_selection_isolating_detour(params):
    """
    Test the greedy selection algorithm by isolating the detour weight. This was done by choosing two different distances to the chargers to create a clear detour difference, and by setting the price to neutralize the side effect that the distance change caused. The expected result is that the algorithm should select the charger with the shorter detour if both price contributions to the score are equal.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 2.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=200.0)
    graph.add_edge(0, 2, distance=400.0)
    graph.add_edge(1, 3, distance=300.0)
    graph.add_edge(2, 3, distance=300.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.2,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 3, params, gparams)
    assert result.feasible
    assert result.route == [0, 1, 3]
    assert result.charge_stops[0]["cost"] == pytest.approx(25.0)


def test_greedy_selection_isolating_charge_rate(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 75.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=300.0)
    graph.add_edge(0, 2, distance=300.0)
    graph.add_edge(1, 3, distance=300.0)
    graph.add_edge(2, 3, distance=300.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 3, params, gparams)
    assert result.feasible
    assert result.route == [0, 2, 3]
    assert result.charge_stops[0]["cost"] == pytest.approx(15.0)


def test_greedy_selection_with_multiple_decisions(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.30, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": True, "price_per_kwh": 0.30, "charge_rate_kw": 50.0}),
            (4, {"name": "Node 4", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (5, {"name": "Node 5", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=300.0)
    graph.add_edge(0, 2, distance=300.0)
    graph.add_edge(1, 3, distance=300.0)
    graph.add_edge(2, 4, distance=300.0)
    graph.add_edge(3, 5, distance=300.0)
    graph.add_edge(4, 5, distance=300.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 5, params, gparams)
    assert result.feasible
    assert result.route == [0, 1, 3, 5]
    assert len(result.charge_stops) == 2
    assert result.charge_stops[0]["cost"] == pytest.approx(18.0)
    assert result.charge_stops[1]["cost"] == pytest.approx(9.0)


def test_greedy_selection_with_unreachable_target():
    graph = nx.DiGraph()
    graph.add_node(0)
    result = greedy_route(graph, 0, 1)
    assert not result.feasible
    assert result.route == []
    assert result.charge_stops == []
    assert result.total_weight == inf
    assert result.total_cost_dollars == pytest.approx(0.0)
    assert result.total_time_h == pytest.approx(0.0)
    assert result.total_distance_km == pytest.approx(0.0)


def test_greedy_selection_with_no_path_to_target(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": False}),
        ]
    )

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 1, params, gparams)
    assert not result.feasible
    assert result.route == []
    assert result.charge_stops == []
    assert result.total_weight == inf
    assert result.total_cost_dollars == pytest.approx(0.0)
    assert result.total_time_h == pytest.approx(0.0)
    assert result.total_distance_km == pytest.approx(0.0)


def test_greedy_selection_with_range_gap(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=500.0)

    gparams = GreedyParams(
        w_progress=1.0,
        w_price=1.0,
        w_detour=0.01,
        w_rate=0.001,
    )

    result = greedy_route(graph, 0, 1, params, gparams)
    assert not result.feasible
    assert result.route == [0]
    assert result.charge_stops == []
    assert result.total_weight == inf
    assert result.total_cost_dollars == pytest.approx(0.0)
    assert result.total_time_h == pytest.approx(0.0)
    assert result.total_distance_km == pytest.approx(0.0)


def test_greedy_cost_upper_bound_matches_greedy_route(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.30, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )

    graph.add_edge(0, 1, distance=300.0)
    graph.add_edge(0, 2, distance=300.0)
    graph.add_edge(1, 3, distance=300.0)
    graph.add_edge(2, 3, distance=300.0)

    expected = greedy_cost_upper_bound(graph, 0, 3, params)
    actual = greedy_route(graph, 0, 3, params).total_weight
    assert expected == actual