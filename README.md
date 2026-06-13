## EV Route and Charging Planner

An algorithm-driven route optimizer for electric vehicles that finds the minimum-cost path between two locations while respecting battery range constraints and strategically selecting charging stops along the way.

# Tasks
* Build road network graph from CSV or simulated data
* Add charging station metadata to each node (charge rate, cost)
* Create state space graph
* Add recharge edges with cost weighting
* Run Djikstra's on state space graph
* Path reconstruction to return full routes
* Greedy charging station selector