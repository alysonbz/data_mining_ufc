import numpy as np
import pandas as pd


# Supondo que a função load_pokemon_dataset() carregue um DataFrame do dataset de Pokémon
def load_pokemon_dataset():
    # Placeholder para a função de carregamento do dataset
    # Retorna um DataFrame de exemplo
    return pd.DataFrame({
        'HP': [45, 60, 80, 39, 58, 78],
        'Attack': [49, 62, 82, 52, 64, 84]
    })


df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster, df):
    coord_list = df.sample(n=num_of_cluster).values.tolist()
    label_list = list(range(num_of_cluster))
    return coord_list, label_list


def create_points(df):
    return df.values.tolist()


def dist_euclidian(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def assign_clusters(coords, centroids):
    coord_labels = []
    for coord in coords:
        distances = [dist_euclidian(coord, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        coord_labels.append(min_distance_index)
    return coord_labels


def update_centroids(coords, coord_labels, num_of_cluster):
    new_centroids = []
    for i in range(num_of_cluster):
        cluster_points = [coords[j] for j in range(len(coords)) if coord_labels[j] == i]
        new_centroids.append(np.mean(cluster_points, axis=0).tolist())
    return new_centroids


def kmeans(df, num_of_cluster, max_iterations=100):
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster, df)
    coords = create_points(df)
    coord_labels = [None] * len(coords)

    for _ in range(max_iterations):
        new_coord_labels = assign_clusters(coords, centroids)
        new_centroids = update_centroids(coords, new_coord_labels, num_of_cluster)

        if new_coord_labels == coord_labels:
            break
        centroids = new_centroids
        coord_labels = new_coord_labels

    return coord_labels, centroids


num_of_clusters = 2
labels, centroids = kmeans(df, num_of_clusters)

print("Labels:", labels)
print("Centroids:", centroids)
