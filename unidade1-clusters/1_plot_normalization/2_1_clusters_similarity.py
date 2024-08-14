from src.utils import load_pokemon_dataset
import numpy as np
from scipy.spatial.distance import euclidean


def compute_single_linkage(cluster1, cluster2):
    min_distance = float('inf')
    for point1 in cluster1:
        for point2 in cluster2:
            distance = euclidean(point1, point2)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def compute_complete_linkage(cluster1, cluster2):
    max_distance = 0
    for point1 in cluster1:
        for point2 in cluster2:
            distance = euclidean(point1, point2)
            if distance > max_distance:
                max_distance = distance
    return max_distance


def compute_average_linkage(cluster1, cluster2):
    distances = []
    for point1 in cluster1:
        for point2 in cluster2:
            distances.append(euclidean(point1, point2))
    return np.mean(distances)


def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return euclidean(centroid1, centroid2)


def compute_ward_linkage(cluster1, cluster2):
    combined_cluster = np.vstack((cluster1, cluster2))
    centroid_combined = np.mean(combined_cluster, axis=0)
    sse_combined = np.sum((combined_cluster - centroid_combined) ** 2)

    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    sse1 = np.sum((cluster1 - centroid1) ** 2)
    sse2 = np.sum((cluster2 - centroid2) ** 2)

    return sse_combined - (sse1 + sse2)

cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade pelo método de Ward: ", compute_ward_linkage(cluster1,cluster2))



