from src.utils import load_pokemon_dataset
import numpy as np

# Funções de similaridade implementadas
def compute_single_linkage(cluster1, cluster2):
    min_distancia = float('inf')
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            if distancia < min_distancia:
                min_distancia = distancia
    return min_distancia

def compute_complete_linkage(cluster1, cluster2):
    max_distancia = 0
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            if distancia > max_distancia:
                max_distancia = distancia
    return max_distancia

def compute_average_linkage(cluster1, cluster2):
    total_distancia = 0
    cont = 0
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = np.linalg.norm(np.array(ponto1) - np.array(ponto2))
            total_distancia += distancia
            cont += 1
    return total_distancia / cont

def compute_centroid_linkage(cluster1, cluster2):
    c1 = np.mean(cluster1, axis=0)
    c2 = np.mean(cluster2, axis=0)
    distancia = np.linalg.norm(c1 - c2)
    return distancia

def compute_ward_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    # Calculate the squared Euclidean distance between centroids
    ward_distance = np.sum(np.square(centroid1 - centroid2))
    return ward_distance

# Clusters
cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

# Chame as funções de similaridade
similaridade_single = compute_single_linkage(cluster1, cluster2)
similaridade_complete = compute_complete_linkage(cluster1, cluster2)
similaridade_average = compute_average_linkage(cluster1, cluster2)
similaridade_centroid = compute_centroid_linkage(cluster1, cluster2)
similaridade_ward = compute_ward_linkage(cluster1, cluster2)

# Imprima os resultados
print("Similaridade ligação simples: ", similaridade_single)
print("Similaridade ligação completa: ", similaridade_complete)
print("Similaridade ligação média: ", similaridade_average)
print("Similaridade pelo método do centroide: ", similaridade_centroid)
print("Similaridade ligação Ward: ", similaridade_ward)
