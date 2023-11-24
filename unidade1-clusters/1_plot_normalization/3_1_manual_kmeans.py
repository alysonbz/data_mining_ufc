from src.utils import load_pokemon_dataset
import numpy as np
import pandas as pd
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score




def set_random_cluster_coordinate(df, num_of_clusters):
    indices = np.random.choice(len(df), num_of_clusters, replace=False)
    centroids = df.iloc[indices].values
    label_list = range(0, num_of_clusters)
    return centroids, label_list


def create_points(df): #não se é necessário pois o df já contem os dados
    coords = df  # Já são coordenadas x, y
    return coords


def dist_euclidean(p1, p2): #Calcula a distância euclidiana fazendo a norma da diferença dos arrays
    p1 = np.array(p1)  # Converte para array
    p2 = np.array(p2)  # Converte para array
    return np.linalg.norm(p1 - p2)


def assign_points_to_clusters(coords, centroids): #calcula a distância para cada centro usando a função dist_euclidean,
                                                  # e atribui o rótulo do cluster do centro mais próximo.
    coord_labels = []
    for coord in coords:
        distances = [dist_euclidean(coord, centroid) for centroid in centroids]
        closest_label = np.argmin(distances)
        coord_labels.append(closest_label)
    return coord_labels


def update_centroids(coords, coord_labels, num_of_clusters): #Ela calcula a média das coordenadas de todos os pontos atribuídos a um determinado cluster e adiciona o novo centro à lista
    new_centroids = []
    for i in range(num_of_clusters):
        cluster_points = [coords[j] for j in range(len(coords)) if coord_labels[j] == i]
        if cluster_points:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
    return new_centroids


def kmeans(df, num_of_clusters): #implementação do k-means
    centroids, centroids_labels = set_random_cluster_coordinate(df, num_of_clusters)
    increase_cluster = True
    coords = df.values
    coord_labels = []

    while increase_cluster:
        coord_labels = assign_points_to_clusters(coords, centroids)
        new_centroids = update_centroids(coords, coord_labels, num_of_clusters)

        if np.all(np.isclose(centroids, new_centroids)):
            increase_cluster = False
        centroids = new_centroids

    return coord_labels


# Conjunto de dados fornecido por você
pokemon = np.array([
    [9.0, 8.0],
    [6.0, 4.0],
    [2.0, 10.0],
    [3.0, 6.0],
    [1.0, 0.0],
    [7.0, 4.0],
    [1.0, 10.0],
    [6.0, 10.0],
    [1.0, 6.0],
    [7.0, 1.0],
    [23.0, 29.0],
    [26.0, 25.0],
    [25.0, 30.0],
    [23.0, 29.0],
    [21.0, 29.0],
    [23.0, 30.0],
    [23.0, 25.0],
    [20.0, 27.0],
    [30.0, 26.0],
    [23.0, 30.0]
])

column_names = ['x', 'y']
df = pd.DataFrame(data=pokemon, columns=column_names)

num_of_clusters = 2
resulting_labels = kmeans(df, num_of_clusters)
print(resulting_labels)