"""
This module loads the road network graph from CSV files and provides a summary of its properties.
"""

import pandas as pd
import networkx as nx
# import matplotlib.pyplot as plt

NODES = pd.read_csv('data/nodes.csv')
EDGES = pd.read_csv('data/edges.csv')

def load_graph(nodes, edges):

    G = nx.DiGraph()

    for _, row in nodes.iterrows():
        G.add_node(row['node_id'],
                   name=row['City'],
                   latitude=row['Latitude'],
                   longitude=row['Longitude'],
                   is_charger=False)

    for _, row in edges.iterrows():
        G.add_edge(row['from_id'], row['to_id'],
                   distance=row['distance_km'])
        if row['is_bidirectional']:
            G.add_edge(row['to_id'], row['from_id'],
                       distance=row['distance_km'])

    return G


def print_graph_summary(G):
    distances = list(nx.get_edge_attributes(G, 'distance').values())

    print(f"Nodes      : {len(G.nodes)}")
    print(f"Edges      : {len(G.edges)}")
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
    print_graph_summary(G)
    # show_graph(NODES, G)