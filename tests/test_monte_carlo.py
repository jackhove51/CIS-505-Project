"""
Focused tests for monte_carlo.py.

The project ships as plain scripts (no package, no installed entry point) and
``load_graph`` reads ``data/*.csv`` relative to the current working directory, so
this module puts the project root on ``sys.path`` for imports and the ``graph``
fixture loads the graph with the cwd pinned to that root.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest

from ev_model import VehicleParams
from dijkstra_ev import dijkstra_ev
from load_graph import load_graph
from monte_carlo import (
    NoiseParams,
    _consumption_factor,
    monte_carlo_route,
    simulate_trip,
)

SOURCE, TARGET = 0, 4  # Chicago -> Cleveland: reachable, with range slack to spare


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def params():
    return VehicleParams()


@pytest.fixture(scope="session")
def graph():
    # load_graph() resolves data/*.csv relative to cwd; pin it to the project root.
    prev = os.getcwd()
    os.chdir(_PROJECT_ROOT)
    try:
        return load_graph()
    finally:
        os.chdir(prev)


@pytest.fixture(scope="session")
def route(graph, params):
    result = dijkstra_ev(graph, SOURCE, TARGET, params)
    assert result.reached, "test route 0 -> 4 should be feasible"
    return result


# --------------------------------------------------------------------------- #
# 1. Determinism                                                                #
# --------------------------------------------------------------------------- #
def test_same_seed_is_identical(route, params):
    noise = NoiseParams(sigma=0.15)
    a = monte_carlo_route(route, params, noise, n_trials=1000, seed=7)
    b = monte_carlo_route(route, params, noise, n_trials=1000, seed=7)
    # MonteCarloResult is a plain dataclass, so == compares every field.
    assert a == b


def test_different_seed_changes_draws(route, params):
    noise = NoiseParams(sigma=0.15)  # ~80% feasible -> outcomes vary across seeds
    a = monte_carlo_route(route, params, noise, n_trials=1000, seed=0)
    b = monte_carlo_route(route, params, noise, n_trials=1000, seed=1)
    assert a != b
    # The stochastic signal is feasibility/failure counts (cost & time are fixed).
    assert (a.n_feasible, a.failed_leg_counts) != (b.n_feasible, b.failed_leg_counts)


# --------------------------------------------------------------------------- #
# 2. Zero-noise sanity                                                          #
# --------------------------------------------------------------------------- #
def test_zero_noise_factor_is_exactly_one():
    rng = np.random.default_rng(0)
    assert _consumption_factor(NoiseParams(sigma=0.0), rng) == 1.0


def test_zero_noise_all_feasible_and_cost_matches_plan(route, params):
    noise = NoiseParams(sigma=0.0)

    # Every trial must survive: at the nominal multiplier the continuous-kWh
    # consumption never exceeds the ceil-rounded consumption the search planned.
    mc = monte_carlo_route(route, params, noise, n_trials=500, seed=0)
    assert mc.n_feasible == mc.n_trials
    assert mc.feasibility_prob == 1.0

    # Realized cost is the deterministic planned cost (no spread).
    for key in ("p50", "p90", "p95"):
        assert mc.cost_percentiles[key] == pytest.approx(route.total_cost_dollars)

    # And the same holds at the single-trip level, including the realized time.
    rng = np.random.default_rng(0)
    trip = simulate_trip(route.itinerary, params, noise, rng)
    assert trip.feasible
    assert trip.failed_leg_index is None
    assert trip.realized_cost_dollars == pytest.approx(route.total_cost_dollars)
    assert trip.realized_time_h == pytest.approx(route.total_time_h)


# --------------------------------------------------------------------------- #
# 3. Monotonicity: feasibility is non-increasing in sigma                       #
# --------------------------------------------------------------------------- #
def test_feasibility_non_increasing_in_sigma(route, params):
    low = monte_carlo_route(route, params, NoiseParams(sigma=0.05), n_trials=2000, seed=0)
    high = monte_carlo_route(route, params, NoiseParams(sigma=0.40), n_trials=2000, seed=0)
    assert low.feasibility_prob >= high.feasibility_prob
    # Sanity: the higher sigma must actually bite, else the comparison is vacuous.
    assert high.feasibility_prob < 1.0


# --------------------------------------------------------------------------- #
# 4. Failure accounting                                                         #
# --------------------------------------------------------------------------- #
def test_failure_counts_reconcile_and_index_drive_legs(route, params):
    mc = monte_carlo_route(route, params, NoiseParams(sigma=0.20), n_trials=2000, seed=0)

    n_failed = sum(mc.failed_leg_counts.values())
    assert mc.n_feasible + n_failed == mc.n_trials
    assert n_failed > 0, "sigma=0.20 should produce some failures to test against"

    for idx in mc.failed_leg_counts:
        assert 0 <= idx < len(route.itinerary)
        assert route.itinerary[idx]["action"] == "drive"


# --------------------------------------------------------------------------- #
# 5. Lognormal multiplier has unit mean                                         #
# --------------------------------------------------------------------------- #
def test_consumption_factor_mean_is_one():
    rng = np.random.default_rng(12345)
    noise = NoiseParams(sigma=0.15)
    factors = np.array([_consumption_factor(noise, rng) for _ in range(100_000)])
    assert factors.mean() == pytest.approx(1.0, abs=0.01)  # within 1%
