import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_pokemon_dataset

df = load_pokemon_dataset()
df = pd.DataFrame({'x': df[0], 'y': df[1]})

# Recebe um DataFrame e o número de clusters desejado, e devolve os centroides e suas respectivas labels
def set_random_cluster_coordinate(df, num_of_cluster):
    indices = np.random.choice(len(df), num_of_cluster, replace=False)
    coords = df.iloc[indices, [0, 1]].values
    labels = np.arange(num_of_cluster)
    return coords, labels


# Retorna uma lista de coordenadas a partir do DataFrame
def create_points(df):
    if 'x' in df.columns and 'y' in df.columns:
        return df[['x', 'y']].values
    else:
        raise ValueError("As colunas 'x' e 'y' não estão presentes no DataFrame.")

# Calcula a distância euclidiana entre dois pontos
def dist_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


# Atribui cada ponto ao centróide mais próximo
def set_closest_centroid(coords, centroids):
    coord_labels = np.argmin(np.array([[dist_euclidean(coord, centroid) for centroid in centroids] for coord in coords]), axis=1)
    return coord_labels


# Atualiza os centroides com base nos pontos de dados atribuídos
def set_new_centroids(coords, coord_labels, num_of_cluster):
    new_centroids = np.array([np.mean(coords[coord_labels == cluster_i], axis=0) for cluster_i in range(num_of_cluster) if np.any(coord_labels == cluster_i)])
    return new_centroids


def kmeans(df, num_of_cluster):
    centroids, centroids_labels = set_random_cluster_coordinate(df, num_of_cluster)
    coords = create_points(df)

    increase_cluster = True

    while increase_cluster:
        coord_labels = set_closest_centroid(coords, centroids)
        new_centroids = set_new_centroids(coords, coord_labels, num_of_cluster)

        if np.allclose(centroids, new_centroids):
            increase_cluster = False

        centroids = new_centroids

    return coord_labels, centroids

# Número de clusters desejado
num_of_clusters = 2

# Obtendo os rótulos dos clusters
result_labels, result_centroids = kmeans(df, num_of_clusters)

# Imprimindo os rótulos dos clusters
print(result_labels)

# Plotando os pontos de dados
plt.scatter(df['x'], df['y'], c=result_labels, cmap='viridis', edgecolor='k', s=50)
# Plotando os centroides
plt.scatter(result_centroids[:, 0], result_centroids[:, 1], c='red', marker='X', s=100, label='Centroides')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Resultado do K-means')
plt.legend()
plt.show()
