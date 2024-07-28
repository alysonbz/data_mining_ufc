import numpy as np


def compute_single_linkage(cluster1, cluster2):
    min_distance = float('inf')
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
            if distance < min_distance:
                min_distance = distance
    return min_distance


def compute_complete_linkage(cluster1, cluster2):
    max_distance = float('-inf')
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
            if distance > max_distance:
                max_distance = distance
    return max_distance


def compute_average_linkage(cluster1, cluster2):
    total_distance = 0
    num_pairs = 0
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
            total_distance += distance
            num_pairs += 1
    return total_distance / num_pairs


def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))


def compute_ward_linkage(cluster1, cluster2):
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    mean1 = np.mean(cluster1, axis=0)
    mean2 = np.mean(cluster2, axis=0)

    ss1 = np.sum((cluster1 - mean1) ** 2)
    ss2 = np.sum((cluster2 - mean2) ** 2)

    combined_cluster = np.concatenate((cluster1, cluster2), axis=0)
    mean_combined = np.mean(combined_cluster, axis=0)
    ss_combined = np.sum((combined_cluster - mean_combined) ** 2)

    ward_distance = ss_combined - ss1 - ss2
    return ward_distance


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

print("Similaridade ligação simples: ", compute_single_linkage(cluster1, cluster2))
print("Similaridade ligação completa: ", compute_complete_linkage(cluster1, cluster2))
print("Similaridade ligação média: ", compute_average_linkage(cluster1, cluster2))
print("Similaridade pelo método do centróide: ", compute_centroid_linkage(cluster1, cluster2))
print("Similaridade ligação pelo método de Ward: ", compute_ward_linkage(cluster1, cluster2))
