from src.utils import load_pokemon_dataset
import numpy as np
import random

df = load_pokemon_dataset()

# Verifica o tipo do objeto retornado
print(f"Tipo do dataset carregado: {type(df)}")
if isinstance(df, tuple):
    data_points = df[0]
    print(f"Primeiro elemento do dataset: {type(df[0])}")
    print(f"Conteúdo: {df[0]}")
elif isinstance(df, list):
    data_points = df
    print(f"Primeiro elemento da lista: {type(df[0])}")
    print(f"Conteúdo: {df[0]}")
else:
    raise TypeError("Estrutura desconhecida")


def initialize_centroids(num_of_clusters, data_points):
    min_val = min(data_points)
    max_val = max(data_points)
    # Escolhe centróides aleatórios com uma distribuição mais inteligente
    centroids = random.sample(data_points, num_of_clusters)
    return [[centroid] for centroid in centroids]


def create_points(data_points):
    coords = [[point] for point in data_points]
    return coords


def dist_euclidian(p1, p2):
    return np.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))


def kmeans(data_points, num_of_clusters):
    centroids = initialize_centroids(num_of_clusters, data_points)
    coords = create_points(data_points)
    prev_centroids = [None] * num_of_clusters
    coord_labels = [None] * len(coords)
    max_iterations = 100  # Limite máximo de iterações
    iteration = 0

    while iteration < max_iterations:
        new_coord_labels = []
        # Atribui cada ponto ao centróide mais próximo
        for coord in coords:
            distances = [dist_euclidian(coord, centroid) for centroid in centroids]
            new_coord_labels.append(np.argmin(distances))

        # Verifica se houve mudança nos rótulos
        if new_coord_labels == coord_labels:
            break

        coord_labels = new_coord_labels.copy()

        # Recalcula os centróides
        for i in range(num_of_clusters):
            cluster_points = [coords[j] for j in range(len(coords)) if coord_labels[j] == i]
            if cluster_points:
                centroids[i] = np.mean(cluster_points, axis=0)

        iteration += 1

    return coord_labels


# Definindo o número de clusters
num_of_clusters = 2

# Executando o algoritmo K-means
result_labels = kmeans(data_points, num_of_clusters)

print("Rótulos dos pontos:", result_labels)
