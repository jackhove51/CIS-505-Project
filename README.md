# EV Route and Charging Planner

An algorithm-driven route optimizer for electric vehicles that finds the minimum-cost path between two locations while respecting battery range constraints and strategically selecting charging stops along the way.

## Tasks
* Build road network graph from CSV or simulated data
* Add charging station metadata to each node (charge rate, cost)
* Create state space graph
* Add recharge edges with cost weighting
* Run Djikstra's on state space graph
* Path reconstruction to return full routes
* Greedy charging station selector

## Range-Uncertainty Monte Carlo (T9/T10)

`monte_carlo.py` stress-tests a route that Dijkstra has already planned. It *replays* a `SearchResult.itinerary` (it never re-runs the search), tracking the battery as a continuous value in kWh rather than the quantized SoC levels the search rounds up with, and perturbs each drive leg's consumption with lognormal noise (mean 1.0). This exposes the real range margin that the `ceil`-based search hides — how often the battery actually survives.

Run it:

```
python monte_carlo.py     # plan route 0 -> 4, print the Monte Carlo + robustness report
pytest tests/             # 7 focused tests: determinism, zero-noise, monotonicity, accounting
```

Key functions:
* `simulate_trip(itinerary, params, noise, rng)` — replay one itinerary (fresh noise per drive leg); returns feasibility, final SoC, realized cost/time, and the first failing leg
* `monte_carlo_route(result, params, noise, n_trials, seed)` — aggregate trials into a feasibility probability, cost/time percentiles (p50/p90/p95 over feasible trials), and a per-leg failure histogram
* `robustness_score(mc_result, params)` — collapse a run into one score that rewards feasibility and penalizes cost spread
* `compare_routes(G, source, target, ...)` — replay the same planned route across several noise sigmas to show how robustness degrades as uncertainty grows

Headline finding: route 0 -> 4 (Chicago -> Cleveland) is feasible per Dijkstra, but under sigma = 0.15 noise the battery survives only ~80% of trials, falling to ~65% at sigma = 0.30. Every failure lands on the final leg, Toledo -> Cleveland — the longest stretch after the last charge.