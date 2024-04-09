import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def compute_single_linkage(cluster1,cluster2):

    min_distance = float('inf')
    for p1 in cluster1:
        for p2 in cluster2:
            distance = euclidean_distance(p1, p2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def compute_complete_linkage(cluster1, cluster2):
    max_distance = 0
    for p1 in cluster1:
        for p2 in cluster2:
            distance = euclidean_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
    return max_distance


def compute_average_linkage(cluster1, cluster2):
    total_distance = sum(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2)
    return total_distance / (len(cluster1) * len(cluster2))

def compute_centroid_linkage(cluster1,cluster2):
    centroid1 = [sum(coord) / len(cluster1) for coord in zip(*cluster1)]
    centroid2 = [sum(coord) / len(cluster2) for coord in zip(*cluster2)]
    return euclidean_distance(centroid1, centroid2)

def compute_ward_linkage(cluster1,cluster2):
    centroid1 = [sum(coord) / len(cluster1) for coord in zip(*cluster1)]
    centroid2 = [sum(coord) / len(cluster2) for coord in zip(*cluster2)]
    merged_cluster = cluster1 + cluster2
    merged_centroid = [sum(coord) / len(merged_cluster) for coord in zip(*merged_cluster)]
    return (len(cluster1) * euclidean_distance(centroid1, merged_centroid) +
            len(cluster2) * euclidean_distance(centroid2, merged_centroid)) / len(merged_cluster)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))
print("similaridade ligação simples: ", compute_ward_linkage(cluster1,cluster2))



