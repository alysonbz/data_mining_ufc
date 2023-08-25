from src.utils import load_pokemon_dataset
import numpy as np

def compute_single_linkage(cluster1,cluster2):
    min_dist = float('inf')

    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = 0
            for i in range(len(ponto1)):
                distancia += (ponto1[i] - ponto2[i]) ** 2
            min_dist = min(min_dist, distancia)

    sim_simples = min_dist ** 0.5
    return sim_simples

def compute_complete_linkage(cluster1, cluster2):
    dist_max = 0
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = 0
            for i in range(len(ponto1)):
                distancia += (ponto1[i] - ponto2[i]) ** 2
            dist_max = max(dist_max, distancia)
    return dist_max ** 0.5

def compute_average_linkage(cluster1, cluster2):
    dist_total = 0
    n_pares = 0

    for ponto1 in cluster1:
        for ponto2 in cluster2:
            distancia = 0
            for i in range(len(ponto1)):
                distancia += (ponto1[i] - ponto2[i]) ** 2
            dist_total += distancia
            n_pares += 1

    media_dist = dist_total / n_pares
    media_sim = media_dist ** 0.5
    return media_sim

def compute_centroid_linkage(cluster1,cluster2):
    centroide = lambda cluster: [sum(coord) / len(cluster) for coord in zip(*cluster)]

    centr1 = centroide(cluster1)
    centr2 = centroide(cluster2)

    distancia = 0
    for i in range(len(centr1)):
        distancia += (centr1[i] - centr2[i]) ** 2

    sim_centroide = distancia ** 0.5
    return sim_centroide

def compute_ward_linkage(cluster1,cluster2):
    centroide = lambda cluster: [sum(coord) / len(cluster) for coord in zip(*cluster)]

    centr1 = centroide(cluster1)
    centr2 = centroide(cluster2)

    media2 = 0
    for i in range(len(centr1)):
        media2 += (centr1[i] - centr2[i]) ** 2

    ward_sim = media2
    return ward_sim



cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade pelo método de Ward: ", compute_ward_linkage(cluster1,cluster2))