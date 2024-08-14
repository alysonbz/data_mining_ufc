import numpy as np
import pandas as pd
from src.utils import load_pokemon_dataset

# Certifique-se de que o dataset é um DataFrame
df = load_pokemon_dataset()
if isinstance(df, tuple):
    df = df[0]  # Se 'df' for um tuple, pegue o primeiro elemento, que deve ser o DataFrame


def set_random_cluster_coordinate(num_of_cluster, df):
    coord_list = []
    for _ in range(num_of_cluster):
        coord = df.sample(n=1).iloc[0].tolist()
        coord_list.append(coord)
    label_list = list(range(num_of_cluster))
    return coord_list, label_list


def create_points(df):
    coords = df.iloc[:, [0, 1]].values.tolist()
    return coords


def dist_euclidian(p1, p2):
    dist = np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    return dist


def kmeans(df, num_of_cluster):
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster, df)
    coords = create_points(df)
    coord_label = [None] * len(coords)

    while True:
        new_coord_label = []
        for coord in coords:
            distances = [dist_euclidian(coord, centroid) for centroid in centroids]
            min_distance_label = centroids_labels[np.argmin(distances)]
            new_coord_label.append(min_distance_label)

        if new_coord_label == coord_label:
            break

        coord_label = new_coord_label

        new_centroids = []
        for label in centroids_labels:
            cluster_points = [coords[i] for i in range(len(coords)) if coord_label[i] == label]
            if cluster_points:
                new_centroid = np.mean(cluster_points, axis=0).tolist()
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[label])

        centroids = new_centroids

    return coord_label


num_of_clusters = 2
labels = kmeans(df, num_of_clusters)
print(labels)
