import numpy as np
import random
import pandas as pd

# Carregando o DataFrame
df = pd.read_csv('C:/Users/Luciana/OneDrive/Documentos/mineracao/data_mining_ufc/unidade1-clusters/datasets/pokemon.csv')

def set_random_cluster_coordinate(num_of_cluster, data):
    coord_list = []
    for _ in range(num_of_cluster):
        coord = data[random.randint(0, len(data)-1)]
        coord_list.append(coord)
    label_list = range(num_of_cluster)
    return coord_list, label_list

def create_points(df):
    coords = df[['x', 'y']].values.tolist()  # Corrigido para values.tolist()
    return coords

def dist_euclidean(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)  # Corrigido para distância euclidiana
    return dist

def kmeans(df, num_of_cluster, max_iterations=100):
    coords = create_points(df)
    centroids, _ = set_random_cluster_coordinate(num_of_cluster, coords)
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(num_of_cluster)]
        for coord in coords:
            distances = [dist_euclidean(coord, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(coord)
        
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Verifique se o cluster não está vazio
                new_centroid = np.mean(cluster, axis=0).tolist()
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(centroids))  # Escolhe aleatoriamente um dos centróides existentes
        
        if np.all(np.array(new_centroids) == np.array(centroids)):
            break
        
        centroids = new_centroids
    
    coord_label = [None] * len(coords)
    for i, cluster in enumerate(clusters):
        for coord in cluster:
            index = coords.index(coord)
            coord_label[index] = i
    
    return coord_label

# Definindo o número de clusters e executando o algoritmo
num_of_clusters = 2
labels = kmeans(df, num_of_clusters)
print(f'labels: {labels}')
