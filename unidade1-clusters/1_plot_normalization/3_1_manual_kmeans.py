from src.utils import load_pokemon_dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = range(0, num_of_cluster)
    for _ in range(num_of_cluster):
        x = np.random.randint(0, 150)
        y = np.random.randint(0, 150)
        coord_list.append([x, y])
    return coord_list, label_list


def dist_euclidian(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def kmeans(data, num_of_cluster):
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coord_labels = [None] * len(data[0])

    while increase_cluster:
        increase_cluster = False
        for i, (x, y) in enumerate(zip(data[0], data[1])):
            min_dist = float('inf')
            for j, (centroid_x, centroid_y) in enumerate(centroids):
                dist = dist_euclidian((x, y), (centroid_x, centroid_y))
                if dist < min_dist:
                    min_dist = dist
                    coord_labels[i] = centroids_labels[j]
        new_centroids = []
        new_centroids_labels = []
        for label in set(coord_labels):
            cluster_coords = [(x, y) for (x, y), l in zip(zip(data[0], data[1]), coord_labels) if l == label]
            if len(cluster_coords) == 0:
                continue
            centroid = np.mean(cluster_coords, axis=0)
            new_centroids.append(centroid)
            new_centroids_labels.append(label)
        if len(new_centroids) != len(centroids):
            increase_cluster = True
        centroids = new_centroids
        centroids_labels = new_centroids_labels

    return coord_labels


# Carregar o conjunto de dados
data = load_pokemon_dataset()

# Executar KMeans com 2 clusters
num_of_clusters = 2
cluster_labels = kmeans(data, num_of_clusters)

# Criar um DataFrame do pandas para os dados
df = pd.DataFrame({'Attack': data[0], 'Defense': data[1], 'cluster_labels': cluster_labels})

# Plotar os dados
sns.scatterplot(data=df, x='Attack', y='Defense', hue='cluster_labels')
plt.title('KMeans Clustering')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()
