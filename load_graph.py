"""
This module loads the road network graph from CSV files and provides a summary of its properties.
"""

import pandas as pd
import networkx as nx
# import matplotlib.pyplot as plt

NODES = pd.read_csv('data/nodes.csv')
EDGES = pd.read_csv('data/edges.csv')
CHARGERS = pd.read_csv('data/chargers.csv')

def load_graph(nodes, edges):

    G = nx.DiGraph()

    for _, row in nodes.iterrows():
        G.add_node(row['node_id'],
                   name=row['City'],
                   latitude=row['Latitude'],
                   longitude=row['Longitude'],
                   is_charger=False,
                   charge_rate_kw=None,
                   price_per_kwh=None)

    for _, row in edges.iterrows():
        G.add_edge(row['from_id'], row['to_id'],
                   distance=row['distance_km'])
        if row['is_bidirectional']:
            G.add_edge(row['to_id'], row['from_id'],
                       distance=row['distance_km'])

    return G


def add_chargers(G, chargers):
    for _, row in chargers.iterrows():
        nid = row['node_id']
        if nid not in G:
            print(f"Warning: node_id {nid} not in graph, skipping.")
            continue
        G.nodes[nid]['is_charger'] = True
        G.nodes[nid]['charge_rate_kw'] = row['charge_rate_kw']
        G.nodes[nid]['price_per_kwh'] = row['price_per_kwh']


def print_graph_summary(G):
    distances = list(nx.get_edge_attributes(G, 'distance').values())

    charger_count = sum(1 for _, d in G.nodes(data=True) if d.get('is_charger'))
    print(f"Nodes      : {len(G.nodes)}")
    print(f"Edges      : {len(G.edges)}")
    print(f"Chargers   : {charger_count}")
    print(f"Min dist   : {min(distances):.1f} km")
    print(f"Max dist   : {max(distances):.1f} km")
    print(f"Avg dist   : {sum(distances)/len(distances):.1f} km")


# def show_graph(nodes, G):
#     pos = {row['node_id']: (row['Longitude'], row['Latitude']) 
#        for _, row in nodes.iterrows()}

#     nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=500, font_size=7, arrows=False)
#     plt.title("Michigan, Illinois, Indiana & Ohio road network")
#     plt.show()


if __name__ == "__main__":
    G = load_graph(NODES, EDGES)
    add_chargers(G, CHARGERS)
    print_graph_summary(G)
    # show_graph(NODES, G)