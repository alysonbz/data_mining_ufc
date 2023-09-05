import random
import numpy as np
from src.utils import load_pokemon_dataset
import pandas as pd
import matplotlib.pyplot as plt
df = load_pokemon_dataset()
x_values, y_values = load_pokemon_dataset()

df = pd.DataFrame({'x': x_values, 'y': y_values})

class KMeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self,X):

        random_index = random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X,cluster_group)
            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_group

    def assign_clusters(self,X):
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self,X,cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)

X = df[['x', 'y']].values
cluster_labels = kmeans.fit_predict(X)
df['cluster'] = cluster_labels

clustered_data = df[['x', 'y', 'cluster']]
# Separar os dados por cluster
cluster_data = [clustered_data[clustered_data['cluster'] == i] for i in range(num_clusters)]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(cluster_data[i]['x'], cluster_data[i]['y'], label=f'Cluster {i}', c=colors[i])

# Plotar os centroides
centroids = kmeans.centroids
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], marker='X', s=200, c='black', label=f'Centroide {i}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means')
plt.legend()
plt.show()
