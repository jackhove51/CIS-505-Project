import networkx as nx
import pytest

from dijkstra_ev import dijkstra_ev
from ev_model import VehicleParams


@pytest.fixture
def params():
    return VehicleParams(
        battery_capacity_kwh=80.0,
        soc_levels=20,
        value_of_time_per_hour=20.0,
        consumption_kwh_per_km=0.2,
        reserve_soc_frac=0.1,
    )


def test_dijkstra_ev_with_feasible_route(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": False}),
        ]
    )
    graph.add_edge(0, 1, distance=100.0)
    graph.add_edge(1, 2, distance=300.0)
    result = dijkstra_ev(graph, 0, 2, params)
    assert result.reached
    assert result.route == [0, 1, 2]
    assert result.total_cost_dollars == pytest.approx(4.0)


def test_dijkstra_ev_with_infeasible_route_no_chargers(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": False}),
            (2, {"name": "Node 2", "is_charger": False}),
        ]
    )
    graph.add_edge(0, 1, distance=300.0)
    graph.add_edge(1, 2, distance=100.0)
    result = dijkstra_ev(graph, 0, 2, params)
    assert not result.reached
    assert result.total_weight == float("inf")
    assert result.total_cost_dollars == 0.0
    assert result.total_time_h == 0.0
    assert result.route == []
    assert result.itinerary == []


def test_dijkstra_ev_with_infeasible_route_insufficient_charge(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": False})
        ]
    )
    graph.add_edge(0, 1, distance=500.0)
    result = dijkstra_ev(graph, 0, 1, params)
    assert not result.reached
    assert result.total_weight == float("inf")
    assert result.total_cost_dollars == 0.0
    assert result.total_time_h == 0.0
    assert result.route == []
    assert result.itinerary == []


def test_dijkstra_ev_with_source_as_target(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
        ]
    )
    result = dijkstra_ev(graph, 0, 0, params)
    assert result.reached
    assert result.route == [0]
    assert result.total_cost_dollars == 0.0
    assert result.total_time_h == 0.0
    assert result.itinerary == []
    assert result.total_weight == 0.0
    assert result.states_settled == 0


def test_dijkstra_ev_with_disconnected_graph(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": False}),
        ]
    )
    result = dijkstra_ev(graph, 0, 1, params)
    assert not result.reached
    assert result.total_weight == float("inf")
    assert result.total_cost_dollars == 0.0
    assert result.total_time_h == 0.0
    assert result.route == []
    assert result.itinerary == []
    assert result.states_settled == 1


def test_dijkstra_ev_with_cost_upper_bound_blocks_route(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": False}),
        ]
    )
    graph.add_edge(0, 1, distance=100.0)
    graph.add_edge(1, 2, distance=300.0)

    unbounded = dijkstra_ev(graph, 0, 2, params)
    assert unbounded.reached  # sanity check: route is feasible without a bound

    tight_bound = unbounded.total_weight - 1.0
    bounded = dijkstra_ev(graph, 0, 2, params, cost_upper_bound=tight_bound)
    assert not bounded.reached


def test_dijkstra_ev_with_multiple_charging_stops(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": False}),
            (1, {"name": "Node 1", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (2, {"name": "Node 2", "is_charger": True, "price_per_kwh": 0.40, "charge_rate_kw": 50.0}),
            (3, {"name": "Node 3", "is_charger": False}),
        ]
    )
    graph.add_edge(0, 1, distance=100.0)
    graph.add_edge(1, 2, distance=300.0)
    graph.add_edge(2, 3, distance=300.0)

    result = dijkstra_ev(graph, 0, 3, params)
    assert result.reached
    assert result.route == [0, 1, 2, 3]

    charge_actions = [step for step in result.itinerary if step["action"] == "charge"]
    assert len(charge_actions) == 2
    assert charge_actions[0]["at"] == "Node 1"
    assert charge_actions[1]["at"] == "Node 2"


def test_dijkstra_ev_skips_charging_when_already_full(params):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"name": "Node 0", "is_charger": True, "price_per_kwh": 0.50, "charge_rate_kw": 50.0}),
            (1, {"name": "Node 1", "is_charger": False}),
        ]
    )
    graph.add_edge(0, 1, distance=10.0)

    result = dijkstra_ev(graph, 0, 1, params)
    assert result.reached
    assert all(step["action"] != "charge" for step in result.itinerary)