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
    return minima_distancia

def compute_complete_linkage(cluster1, cluster2):
    # Definindo a menor distancia como infinito
    maior_distancia = float('-inf')
    # Loop para percorrer os dois clusters
    for ponto_1 in cluster1:
        for ponto_2 in cluster2:
            distancia = distancia_euclediana(ponto_1, ponto_2)
            if distancia > maior_distancia:
                maior_distancia = distancia
    return maior_distancia


def compute_average_linkage(cluster1, cluster2):
    for ponto_1 in cluster1:
        for ponto_2 in cluster2:
            distancia = distancia_euclediana(ponto_1, ponto_2)
            media_das_distancias = distancia.mean()
    return media_das_distancias

def compute_centroid_linkage(cluster1,cluster2):
    media_cluss_1 = cluster1.mean()
    media_cluss_2 = cluster2.mean()
    distancia = distancia_euclediana(media_cluss_1, media_cluss_2)

    return distancia

def compute_ward_linkage(cluster1,cluster2):
    return None


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1, cluster2))

'''print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))
'''

