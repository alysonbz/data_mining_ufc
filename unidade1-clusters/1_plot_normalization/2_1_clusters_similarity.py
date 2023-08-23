import numpy as np
from scipy.spatial.distance import euclidean


# Calcula todas as distancias e identifica qual a distancia minima
def compute_single_linkage(cluster1,cluster2):
    distancia_min = None
    for p1 in cluster1:
        for p2 in cluster2:
            distancia = euclidean(p1, p2)
            if distancia_min is None or distancia < distancia_min:
                distancia_min = distancia
    return distancia_min


# Calcula todas as distancias e identifica qual a distancia maxima
def compute_complete_linkage(cluster1, cluster2):
    distancia_max = None
    for p1 in cluster1:
        for p2 in cluster2:
            distancia = euclidean(p1, p2)
            if distancia_max is None or distancia > distancia_max:
                distancia_max = distancia
    return distancia_max


# Calcula a media das distancias
def compute_average_linkage(cluster1, cluster2):
    distancia_total, num_pares = 0, 0
    for p1 in cluster1:
        for p2 in cluster2:
            distancia_total += euclidean(p1, p2)
            num_pares += 1
    return distancia_total / num_pares


# Calcula a distancia das medias
def compute_centroid_linkage(cluster1,cluster2):
    c1 = np.mean(cluster1, axis=0)
    c2 = np.mean(cluster2, axis=0)
    return euclidean(c1, c2)


# Calcula a distancia de cada ponto ao centroide
def compute_ward_linkage(cluster1,cluster2):
    cluster_mesclado = np.vstack((cluster1, cluster2))
    distancia_total = 0
    centroide = np.mean(cluster_mesclado, axis=0)
    for p in cluster_mesclado:
        distancia_total += euclidean(p, centroide) ** 2
    return distancia_total


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))



