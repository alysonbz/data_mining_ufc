
import numpy as np

def dist_euclidiana(coordenada,coordenada2):
    return np.sqrt((coordenada[0] - coordenada2[0]) ** 2 + (coordenada[1] - coordenada2[1]) ** 2)

def centroid(cluster):
    x = sum(p[0] for p in cluster)
    y = sum(p[1] for p in cluster)
    centroid = [x / len(cluster), y / len(cluster)]
    return centroid

def compute_single_linkage(cluster1,cluster2):
    menor_dist = float('infinity')

    for coordenada in cluster1:
        for coordenada2 in cluster2:
            distancia = dist_euclidiana(coordenada,coordenada2)
            if distancia <= menor_dist:
                menor_dist = distancia
    return menor_dist

def compute_complete_linkage(cluster1, cluster2):
     maior_dist = 0

     for coordenada in cluster1:
         for coordenada2 in cluster2:
             distancia = dist_euclidiana(coordenada,coordenada2)
             if  distancia >= maior_dist:
                 maior_dist = distancia
     return maior_dist

def compute_average_linkage(cluster1, cluster2):
     total = 0
     media = float()
     for coordenada in cluster1:
         for coordenada2 in cluster2:
             distancia = dist_euclidiana(coordenada,coordenada2)
             total += distancia
             media = total / (len(cluster1)*len(cluster2))
     return media


def compute_centroid_linkage(cluster1,cluster2):
    centroide1 = centroid(cluster1)
    centroide2 = centroid(cluster2)

    return dist_euclidiana(centroide1,centroide2)

def compute_ward_linkage(cluster1,cluster2):
    centroide1 = centroid(cluster1)
    centroide2 = centroid(cluster2)

    n_cluster1 = len(cluster1)
    n_cluster2 = len(cluster2)
    n_total = n_cluster1 + n_cluster2
    dist_quadratica = (n_cluster1 * dist_euclidiana(centroide1,centroide2)**2 +
        (n_cluster2 * dist_euclidiana(centroide1,centroide2))**2) / n_total

    return dist_quadratica



cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]



print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))

