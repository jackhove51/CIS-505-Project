"""
ARCHIVED: This module is archived and should not be used in the current implementation. It was part of an earlier version of the project and has been replaced by a more efficient and modular approach (see dijkstra_ev.py / ev_model.py). The code below is kept for reference purposes only.
"""

__all__ = ['build_state_space']

import networkx as nx
import math
import pandas as pd

from load_graph import load_graph


def build_state_space(graph, max_charge=600, charge_states=101) -> nx.DiGraph:
    g_next = nx.DiGraph()
    step_size = max_charge / (charge_states - 1)

    for node in graph.nodes(data=True):
        node_id = node[0]
        name = node[1]['name']
        is_charger = node[1]['is_charger']
        for charge_level in range(charge_states):
            state_node = (node_id, charge_level)
            g_next.add_node(state_node, name=name, is_charger=is_charger)

    for u, v, data in graph.edges(data=True):
        distance = data['distance']
        cost_step = math.ceil(distance / step_size)

        for charge_level in range(charge_states):
            if charge_level >= cost_step:
                next_charge_level = charge_level - cost_step
                g_next.add_edge((u, charge_level), (v, next_charge_level),
                                 distance=distance, weight=0.0)

    _add_recharge_edges(g_next, graph, max_charge, charge_states, efficiency_kwh_per_km=0.2)

    return g_next


def _add_recharge_edges(g_next, graph, max_charge, charge_states, efficiency_kwh_per_km):
    step_size = max_charge / (charge_states - 1)

    for node in graph.nodes(data=True):
        node_id = node[0]
        is_charger = node[1]['is_charger']
        if not is_charger:
            continue

        price_per_kwh = node[1]['price_per_kwh']
        energy_per_step_kwh = step_size * efficiency_kwh_per_km
        cost_per_step = energy_per_step_kwh * price_per_kwh

        for charge_level in range(charge_states - 1):
            next_charge_level = charge_level + 1
            g_next.add_edge((node_id, charge_level), (node_id, next_charge_level),
                             weight=cost_per_step, recharge_cost=cost_per_step)


if __name__ == "__main__":
    nodes = pd.read_csv('data/nodes.csv')
    edges = pd.read_csv('data/edges.csv')
    chargers = pd.read_csv('data/chargers.csv')

    G = load_graph(nodes, edges, chargers)
    state_space = build_state_space(G)