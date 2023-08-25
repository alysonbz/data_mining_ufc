from src.utils import load_pokemon_dataset
import numpy as np


def compute_single_linkage(cluster1,cluster2):
    min_distancia = float('inf')

    for i in cluster1:
        for j in cluster2:
            distancia = sum((a - b) ** 2 for a, b in zip(i, j)) ** 0.5
            if distancia < min_distancia:
                min_distancia = distancia

    return min_distancia
def compute_complete_linkage(cluster1, cluster2):
    max_distancia = float('-inf')

    for i in cluster1:
        for j in cluster2:
            distancia = sum((a - b) ** 2 for a, b in zip(i, j)) ** 0.5
            if distancia > max_distancia:
                max_distancia = distancia

    return max_distancia

def compute_average_linkage(cluster1, cluster2):
    total_distancia = 0
    conta = 0

    for i in cluster1:
        for j in cluster2:
            total_distancia += sum((a - b) ** 2 for a, b in zip(i, j)) ** 0.5
            conta += 1

    average_distance = total_distancia / conta
    return average_distance


def compute_centroid_linkage(cluster1,cluster2):
    c1 = np.mean(cluster1, axis=0)
    c2 = np.mean(cluster2, axis=0)
    centroid_distance = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
    return centroid_distance


def compute_ward_linkage(cluster1,cluster2):
   return None

cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))



