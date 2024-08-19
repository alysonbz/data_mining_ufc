import numpy as np
import random
from src.utils import carregar_dataset_pokemon

df = carregar_dataset_pokemon()

# Definir função para inicializar centróides aleatórios
def definir_coordenadas_centroides_aleatorias(num_clusters, df):
    coord_list = df.sample(n=num_clusters).values.tolist()
    label_list = list(range(num_clusters))
    return coord_list, label_list

# Definir função para criar pontos a partir do dataframe
def criar_pontos(df):
    coords = df.values.tolist()
    return coords

# Calcular a distância euclidiana entre dois pontos em N dimensões
def distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

# Implementar o algoritmo K-means
def kmeans(df, num_clusters, max_iteracoes=100):
    centroids, labels_centroids = definir_coordenadas_centroides_aleatorias(num_clusters, df)
    coords = criar_pontos(df)
    labels_coords = [None] * len(coords)

    for _ in range(max_iteracoes):
        for i, coord in enumerate(coords):
            distancias = [distancia_euclidiana(coord, centroide) for centroide in centroids]
            centroide_mais_proximo = np.argmin(distancias)
            labels_coords[i] = labels_centroids[centroide_mais_proximo]

        novos_centroids = []
        for label in labels_centroids:
            pontos_no_cluster = [coords[i] for i in range(len(coords)) if labels_coords[i] == label]
            if pontos_no_cluster:
                novo_centroid = np.mean(pontos_no_cluster, axis=0).tolist()
                novos_centroids.append(novo_centroid)
            else:
                novos_centroids.append(random.choice(coords))

        if novos_centroids == centroids:
            break
        centroids = novos_centroids

    return labels_coords, centroids

num_clusters = 3

# Executar K-means
labels, centroids = kmeans(df, num_clusters)
print("Labels:", labels)
print("Centroides:", centroids)
