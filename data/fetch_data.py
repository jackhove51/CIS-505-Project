'''
This file reads the US cities data from a CSV file, filters for cities in Indiana, Illinois, Michigan, and Ohio, and creates a graph where nodes represent cities and edges represent roads between cities that are within a certain distance threshold (200 km in this case). The resulting nodes and edges are saved to separate CSV files for later use in graph analysis.
'''

import pandas as pd
from pathlib import Path
from geopy.distance import geodesic
from itertools import combinations

# Load CSV into a pandas dataframe
csv_path = Path(__file__).parent / 'US_cities_2022.csv'
df = pd.read_csv(csv_path)

# Filter for cities in Indiana, Illinois, Michigan, and Ohio, and create a new dataframe with the relevant columns
filtered_df = df[df["State"].isin(["Illinois", "Indiana", "Michigan", "Ohio"])].reset_index()
cities_df = filtered_df[["City", "State", "Latitude", "Longitude"]]
cities_df['node_id'] = range(len(cities_df))
cities_df = cities_df[cities_df['City'] != 'Evansville']  # Remove Evansville, IN due to proximity to other cities

# Define a distance threshold (in kilometers) for creating edges between cities
threshold_km = 200
edges = []

# This finds the distance between each pair of cities and creates an edge if the distance is less than or equal to the threshold
for (i, city1), (j, city2) in combinations(cities_df.iterrows(), 2):
    dist = geodesic((city1.Latitude, city1.Longitude), (city2.Latitude, city2.Longitude)).km
    if dist <= threshold_km:
        edges.append({
            'from_id': city1.node_id,
            'to_id': city2.node_id,
            'distance_km': round(dist, 2),
            'is_bidirectional': True
        })

edges_df = pd.DataFrame(edges)
print(f"Generated {len(cities_df)} nodes and {len(edges_df)} edges based on a distance threshold of {threshold_km} km.")

cities_df.to_csv('data/nodes.csv', index=False)
edges_df.to_csv('data/edges.csv', index=False)