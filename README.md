# EV Route and Charging Planner

An algorithm-driven route optimizer for electric vehicles that finds the minimum-cost path between two locations while respecting battery range constraints and strategically selecting charging stops along the way.

## Tasks
* Find US cities dataset (`data/US_cities_2022.csv`, `data/fetch_data.py`) 
* Construct nodes and edges dataframes from CSV (`nodes.csv`, `edges.csv`)
* Build chargers dataset/dataframe using nodes/edges data (`chargers.csv`)
* Build road network graph from CSV (`load_graph.py`)
* Attach charging station metadata to graph nodes (charge rate, price)
* Model EV physics and cost formulas as reusable parameters (`ev_model.py`)
* Run Dijkstra's search over the on-the-fly (node, charge level) state space (`dijkstra_ev.py`)
* Reconstruct full routes and itineraries from the search result
* Greedy charging station selector as a fast baseline (`greedy.py`)
* Use the greedy solution as an upper bound to prune Dijkstra's search
* Monte Carlo simulation of range uncertainty on planned routes (`monte_carlo.py`)
* Approximation benchmark: runtime vs accuracy tradeoff across soc_levels discretization (`approximation.py`)
* Automated test coverage (`tests/*.py`)

## Data Preparation & Graph Loading

### Data Sources and Graph Construction

The project builds a road network for the US Midwest (Illinois, Indiana, Michigan, Ohio) from two data sources:

1. **US Cities Dataset** (`data/US_cities_2022.csv`): Contains latitude/longitude for ~30,000 US cities
2. **Chargers Dataset** (`data/chargers.csv`): Curated list of charging stations with rate (kW) and price ($/kWh)

The script `data/fetch_data.py` filters the cities dataset to the four-state region, then builds edges by computing geodesic distances between city pairs; any two cities within 200 km are connected. This yields ~20 nodes (cities) and ~100 edges (roads), saved to `data/nodes.csv` and `data/edges.csv`. The chargers dataset is loaded separately and merged into the graph as node attributes (marking which cities have charging stations and at what price/rate).

Run it:

```
python data/fetch_data.py      # regenerate nodes.csv and edges.csv from US_cities_2022.csv
```

This produces:
* `data/nodes.csv` — node_id, City, State, Latitude, Longitude
* `data/edges.csv` — from_id, to_id, distance_km, is_bidirectional
* `data/chargers.csv` — node_id, charge_rate_kw, price_per_kwh (manually curated for this region)

### Graph Loading and Representation

`load_graph.py` loads the three CSV files into a NetworkX `DiGraph` with enriched node and edge attributes:

```
python load_graph.py           # load and print graph summary
```

Key functions:
* `load_graph()` — returns a `DiGraph` with nodes storing `{name, latitude, longitude, is_charger, charge_rate_kw, price_per_kwh}` and edges storing `{distance}`; cached for efficiency
* `print_graph_summary(G)` — print node/edge counts, charger count, distance statistics

The graph is a foundation for all downstream algorithms (Dijkstra, greedy, Monte Carlo, approximation).

## EV Physics and Cost Model

`ev_model.py` centralizes the EV physics formulas and cost accounting. It defines `VehicleParams`, a dataclass holding battery and operational parameters, and provides pure functions to compute energy consumption, charging cost/time, and driving time.

**VehicleParams** (configurable defaults shown):
* `battery_capacity_kwh=75.0` — total battery energy
* `consumption_kwh_per_km=0.20` — energy per km driven
* `start_soc_frac=1.0` — initial state of charge (fraction of capacity)
* `reserve_soc_frac=0.10` — minimum safe SoC (battery never drains below this)
* `avg_speed_kmh=100.0` — average driving speed
* `soc_levels=20` — discretization: battery state space split into 0..20 levels
* `w_cost=1.0` — cost weight (USD per unit energy or unit time)
* `value_of_time_per_hour=18.0` — USD/hr, roughly half US median wage; used to value driving/charging time

**Key functions**:
* `level_kwh(params)` — energy per SoC level = `battery_capacity / soc_levels`
* `drive_levels(dist_km, params)` — energy cost in SoC levels (always rounds up via `ceil` for safety)
* `drive_weight(dist_km, params)` — time cost in USD via value-of-time
* `charge_step_weight(price, rate_kw, params)` — cost to gain one SoC level = fuel + time-to-charge
* `reserve_level(params)`, `start_level(params)` — convert SoC fractions to discrete levels

The `ceil`-based rounding in `drive_levels()` is critical: it ensures battery feasibility by always rounding up energy demand, but introduces a cost inflation that grows coarser as `soc_levels` shrinks — this tradeoff is the focus of `approximation.py`.

## Dijkstra EV Search

`dijkstra_ev.py` implements Dijkstra's algorithm over a lifted state space `(node, soc_level)` where `soc_level` is the battery's discretized state of charge (0 to `params.soc_levels`). The search explores both driving to neighboring nodes (consuming battery) and charging at stations (incrementing the battery level by one per step), always respecting a reserve margin below which the battery cannot go. Edge costs combine time, fuel, and charging fees via a weighted sum configurable in `VehicleParams`. The algorithm returns a `SearchResult` with the optimal route, itinerary (step-by-step drive and charge actions), total cost/time, and search counters for profiling.

Run it:

```
python dijkstra_ev.py          # plan optimal route 0 -> 4, print itinerary
pytest tests/test_monte_carlo.py  # 7 tests validate the search and itinerary reconstruction
```

Key functions and classes:
* `SearchResult` — dataclass holding the feasibility flag, route (list of nodes), itinerary (list of drive/charge dicts), cost, time, and search metrics (`states_settled`, `heap_pushes`)
* `dijkstra_ev(G, source, target, params, cost_upper_bound)` — run the search; returns a `SearchResult`; accepts an optional `cost_upper_bound` to prune states with cost exceeding the bound (used by greedy-based optimization)
* `print_itinerary(result, G, params)` — pretty-print the route and itinerary with per-step distance, SoC transitions, cost/time, and totals

**Key insight**: By discretizing SoC into `soc_levels` bins and using `ceil`-based rounding in `drive_levels()` (energy per km), the search always rounds UP, ensuring battery feasibility. However, coarser discretizations (fewer SoC levels) waste more battery per leg, inflating the real cost — this is the discretization tradeoff benchmarked in `approximation.py`.

## Greedy Charging Selector

`greedy.py` provides a fast greedy heuristic for EV routing: starting from the source, repeatedly choose the "best" charger within range using a weighted scoring function (progress toward target, fuel price, detour distance, and charge rate), then continue driving. The greedy result is always feasible if any path exists (per-hop `ceil` accounting matches the Dijkstra lifted space) and serves as an upper bound for cost-based pruning in the optimal search.

Run it:

```
python greedy.py            # solve route 0 -> 4 with greedy, then optimal with pruning; compare
pytest tests/ # greedy tests integrated into larger suite
```

Key functions and classes:
* `GreedyParams` — dataclass with weights `w_progress`, `w_price`, `w_detour`, `w_rate` controlling the greedy choice heuristic
* `GreedyResult` — holds the route, charge stops, feasibility flag, and cost/time/distance totals
* `greedy_route(G, source, target, params, gparams)` — run the greedy algorithm; always returns a feasible route if one exists
* `solve_with_pruning(G, source, target, params, gparams)` — run greedy first to get a cost upper bound, then run Dijkstra with that bound to prune the search space; returns a `PruningResult` with both the greedy and optimal solutions and metrics on states/pushes saved
* `print_greedy_result(result, G, params)` — pretty-print the greedy route and charge stops
* `_path_levels(G, u, v, params)` — compute energy needed along the shortest-distance path from u to v (used by the greedy heuristic)

**Key insight**: The greedy's cost often comes within a few percent of optimal (e.g., within 5%), making it a cheap way to prune Dijkstra's search. On some routes the greedy is suboptimal (e.g., choosing high-cost stations early), which is why pruning-enhanced Dijkstra still finds the true optimum.

## Range-Uncertainty Monte Carlo

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

## Approximation Benchmark

`approximation.py` benchmarks the `soc_levels` discretization knob to expose the runtime-vs-accuracy tradeoff. The search uses `soc_levels` to quantize the battery state space (finer grids = more states, longer search time); `ceil`-based rounding in `drive_levels()` wastes battery per leg, inflating the real cost by a factor $(1 + \epsilon)$. This module sweeps `soc_levels` over a list of values and rates each against a fine-grained baseline, reporting the "knee" point where accuracy-per-state is best. Runtime is measured by search counters (`states_settled`, `heap_pushes`), not wall-clock time, ensuring reproducibility.

Run it:

```
python approximation.py           # benchmark route 0 -> 4, print accuracy and knee curve
pytest tests/test_approximation.py # 4 focused tests: non-monotonicity, rerouting, epsilon tracking
```

Key functions:
* `run_approximation(G, source, target, levels_list, baseline_levels, params)` — sweep `soc_levels` over a list and return cost/runtime for each against the baseline ground truth
* `print_approximation_report(result, route_label)` — tabulate the sweep with cost, approximation ratio, and settled state counts; list the chosen path per soc_levels setting
* `print_tradeoff_report(aggregate, baseline_states)` — show the accuracy curve (epsilon vs soc_levels) and identify the knee point (best accuracy-per-state)
* `empirical_epsilon(aggregate)` — map each `soc_levels` to its measured $(1 + \epsilon)$ inflation as a dictionary
* `theoretical_bound_note()` — return a plain-text explanation of why coarsening inflates cost by tracking worst-case wasted charge per leg

Headline finding: the approximation curve is **non-monotonic** — finer grids sometimes settle *fewer* states (e.g., L=16 can beat L=20). This happens when coarser grids trigger structural reroutes that explore different neighborhoods. The "knee" identifies the soc_levels setting that best balances cost inflation against state count reduction; for route 0 -> 4, L=16 offers a strong tradeoff, settling ~35% fewer states than the baseline L=40 while inflating cost by only ~8.7%.

## Automated Test Coverage

The project includes 43 focused unit and integration tests across five test files, validating correctness and properties of each module:

**`tests/test_ev_model.py`** (13 tests)  
Validates EV physics formulas and parameter conversions:
* `level_kwh()` division and rounding behavior
* `reserve_level()` and `start_level()` SoC conversions (ceil vs round)
* `drive_levels()` energy accounting with `ceil` rounding
* `drive_weight()` and `charge_step_weight()` cost formulas
* Parameter consistency (reserve never exceeds soc_levels, battery capacity preserved)

**`tests/test_load_graph.py`** (8 tests)  
Ensures correct graph construction from CSV files:
* CSV dataframes have expected columns (nodes, edges, chargers)
* Graph nodes and edges match CSV row counts
* Charger attributes (rate, price) are correctly attached to nodes
* Invalid charger node IDs (not in the graph) are gracefully skipped

**`tests/test_dijkstra_ev.py`** (8 tests)  
Validates the search algorithm and itinerary reconstruction:
* Feasible routes are found with correct cost/time accounting
* Infeasible routes (unreachable, insufficient battery, no chargers) return `reached=False`
* Edge cases: source==target, disconnected graphs, fully charged source skips charging
* Cost upper bound pruning works correctly (rejects states exceeding bound)
* Multiple charging stops are correctly sequenced

**`tests/test_monte_carlo.py`** (7 tests)  
Validates the Monte Carlo simulation framework and robustness properties:
* Same seed produces identical trials (determinism)
* Zero noise (sigma=0) yields feasibility 100% and cost matching the plan
* Feasibility degrades monotonically as noise increases
* Failure counts reconcile with feasibility totals; failures index correct drive legs
* Lognormal consumption factor has mean 1.0

**`tests/test_approximation.py`** (7 tests)  
Validates the discretization benchmark and cost-accuracy tradeoff:
* Baseline soc_levels=40 ratio is exactly 1.0 (ground truth)
* Coarser grids never outperform the fine baseline (cost lower bound)
* States settled is monotonic in soc_levels (finer = more states)
* On average, coarser grids have worse accuracy than fine grids
* Coarse grids (L<=8) structurally reroute; fine grids (L>=10) match the baseline route
* Empirical epsilon (cost inflation) matches the computed approximation ratio

Run all tests:

```
pytest tests/                  # run all 43 tests with verbose output
pytest -v tests/               # show individual test names
pytest tests/test_monte_carlo.py -k "sigma" # run tests matching keyword
```