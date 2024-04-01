from src.utils import load_pokemon_dataset
import numpy as np

p1 = 1
p2 = 7

def distancia_euclediana(ponto_1, ponto_2):
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(ponto_1, ponto_2)))

def compute_single_linkage(cluster1,cluster2):
    #Definindo a menor distancia como infinito
    minima_distancia = float('inf')
    #Loop para percorrer os dois clusters
    for ponto_1 in cluster1:
        for ponto_2 in cluster2:
            distancia = distancia_euclediana(ponto_1, ponto_2)
            if distancia < minima_distancia:
                minima_distancia = distancia
    return distancia

def compute_complete_linkage(cluster1, cluster2):
     return None

def compute_average_linkage(cluster1, cluster2):
     return None

def compute_centroid_linkage(cluster1,cluster2):
    return None

def compute_ward_linkage(cluster1,cluster2):
    return None


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))

'''
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))
'''

