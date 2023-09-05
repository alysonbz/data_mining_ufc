from src.utils import load_pokemon_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x, y = load_pokemon_dataset()

# distância Euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# incializando os centroides aleatoriamente
def initialize_centroids(num_clusters, data):
    indices = np.random.choice(len(data), num_clusters, replace=False)
    centroids = [data[i] for i in indices]
    return centroids

# atribuindo pontos aos clusters com base nos centroides
def assign_to_clusters(data, centroids):
    cluster_labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_labels.append(np.argmin(distances))
    return cluster_labels

# calculando novos centroides
def update_centroids(data, cluster_labels, num_clusters):
    new_centroids = []
    for i in range(num_clusters):
        cluster_points = [data[j] for j, label in enumerate(cluster_labels) if label == i]
        if cluster_points:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
    return new_centroids

#  convergência
def check_convergence(centroids, new_centroids, tolerance=1e-4):
    return all(np.linalg.norm(np.array(c1) - np.array(c2)) < tolerance for c1, c2 in zip(centroids, new_centroids))

#  K-means
def kmeans(data, num_clusters):
    centroids = initialize_centroids(num_clusters, data)
    converged = False

    while not converged:
        cluster_labels = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_labels, num_clusters)
        converged = check_convergence(centroids, new_centroids)
        centroids = new_centroids

    return cluster_labels, centroids


data = list(zip(x, y))

# Número de clusters
num_clusters = 2

# Execute o K-means
cluster_labels, centroids = kmeans(data, num_clusters)

# Visualização dos clusters
df_result = pd.DataFrame({"X": x, "Y": y, "Cluster": cluster_labels})
plt.scatter(df_result['X'], df_result['Y'], c=df_result['Cluster'])
plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='black', marker='X', s=100, label='Centroides')  # Adicione esta linha para marcar os centroides
plt.title('Agrupamento K-means')
plt.legend(loc='upper right')  # Adicione um rótulo para os centroides
plt.show()