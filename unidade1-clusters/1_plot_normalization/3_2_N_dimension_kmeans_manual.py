import numpy as np
import random
from src.utils import load_pokemon_dataset

# Carregar o dataset (a função `load_pokemon_dataset` é presumida, como parte do contexto fornecido)
df = load_pokemon_dataset()

# Definir função para inicializar centróides aleatórios
def set_random_cluster_coordinate(num_of_cluster, df):
    # Selecionar coordenadas aleatórias do dataset
    coord_list = df.sample(n=num_of_cluster).values.tolist()
    label_list = list(range(num_of_cluster))
    return coord_list, label_list

# Definir função para criar pontos a partir do dataframe
def create_points(df):
    coords = df.values.tolist()
    return coords

# Definir função para calcular a distância euclidiana entre dois pontos em N dimensões
def dist_euclidian(p1, p2):
    dist = np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    return dist

# Implementar o algoritmo K-means para N dimensões
def kmeans(df, num_of_cluster, max_iterations=100):
    # Inicializar centróides aleatórios
    centroids, centroids_labels = set_random_cluster_coordinate(num_of_cluster, df)
    coords = create_points(df)
    coord_label = [None] * len(coords)

    for _ in range(max_iterations):
        # Atribuir cada ponto ao centróide mais próximo
        for i, coord in enumerate(coords):
            distances = [dist_euclidian(coord, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            coord_label[i] = centroids_labels[closest_centroid]

        # Recalcular os centróides
        new_centroids = []
        for label in centroids_labels:
            points_in_cluster = [coords[i] for i in range(len(coords)) if coord_label[i] == label]
            if points_in_cluster:
                new_centroid = np.mean(points_in_cluster, axis=0).tolist()
                new_centroids.append(new_centroid)
            else:
                # Se um cluster perder todos os seus pontos, inicialize-o novamente
                new_centroids.append(random.choice(coords))

        # Verificar convergência
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return coord_label, centroids

# Parâmetros
num_of_clusters = 3  # Ajuste conforme necessário

# Executar K-means
labels, centroids = kmeans(df, num_of_clusters)
print("Labels:", labels)
print("Centroids:", centroids)
