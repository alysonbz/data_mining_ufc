from src.utils import load_pokemon_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Pokemon dataset
x, y = load_pokemon_dataset()

# Data Preparation
def create_data_points(x_cord, y_cord):
    return [[x, y] for x, y in zip(x_cord, y_cord)]

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def initialize_centroids(data, num_clusters):
    random_points = np.random.choice(len(data), num_clusters, replace=False)
    centroids = [data[i] for i in random_points]
    return centroids

def assign_to_clusters(data, centroids):
    distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data])
    return np.argmin(distances, axis=1)

def update_centroids(data, current_clusters, num_clusters):
    updated_centroids = []
    for i in range(num_clusters):
        cluster_points = [data[j] for j in range(len(data)) if current_clusters[j] == i]
        if not cluster_points:
            updated_centroids.append([0, 0])
        else:
            cluster_mean = np.mean(cluster_points, axis=0)
            updated_centroids.append(cluster_mean)
    return updated_centroids

def k_means_clustering(df, num_clusters):
    centroids = initialize_centroids(df, num_clusters)

    while True:
        current_clusters = assign_to_clusters(df, centroids)

        new_centroids = update_centroids(df, current_clusters, num_clusters)

        if np.all([np.array(centroids[i]) == np.array(new_centroids[i]) for i in range(num_clusters)]):
            break
        centroids = new_centroids

    return current_clusters, centroids

# Visualize the data
plt.scatter(x, y)

# Prepare the data
data = create_data_points(x, y)
num_of_clusters = 3

# Perform K-means clustering
clusters, cluster_centers = k_means_clustering(data, num_of_clusters)

# Create a DataFrame with cluster assignments
data_final = pd.DataFrame({"X": [p[0] for p in data], "Y": [p[1] for p in data], "Cluster": clusters})

# Visualize the clustered data
plt.scatter(data_final['X'], data_final['Y'], c=data_final['Cluster'])
plt.show()
