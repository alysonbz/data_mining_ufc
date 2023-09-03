from src.utils import load_pokemon_dataset
import numpy as np


np.random.seed(12)

x, y = load_pokemon_dataset()


# Preecher a lista com quatro coordenadas aleatótias
## Recebe as listas de coordenadas e o numero de clusters desejado e devolve os centroides e suas respectivas labels
def set_random_cluster_coordinate(x, y, num_of_cluster):
    indices = np.random.choice(len(x), num_of_cluster, replace=False)
    coords = np.array([[x[i], y[i]] for i in indices])
    labels = np.arange(num_of_cluster)
    return coords, labels


# Retorna uma lista de coordenadas
def create_points(x, y):
    return np.array(list(zip(x, y)))


# calcula a distancia euclidiana entre dois pontos
def dist_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def kmeans(x, y, num_of_cluster):
    centroids, centroids_labels = set_random_cluster_coordinate(x, y, num_of_cluster) # define 4 pontos centroids aleatorios
    coords = create_points(x, y) # transforma x e y em coordenadas
    increase_cluster = True

    while increase_cluster:
        coord_labels = []

        # Atribui cada pontos ao centróide mais próximo
        for coord in coords:
            distances = [dist_euclidean(coord, centroid) for centroid in centroids] # calcula a distancia entre os pontos
            coord_labels.append(np.argmin(distances)) # guarda o indice da menor distancia (label)

        # Atualizar centroides com base nos pontos de dados atribuídos
        new_centroids = []
        for cluster_i in range(num_of_cluster):
            # cria uma lista com os pontos atribuidos a cada centroide
            cluster_points = [coords[i] for i, label in enumerate(coord_labels) if label == cluster_i]
            if cluster_points: # se tiver pontos no cluster atual
                cluster_mean = np.mean(cluster_points, axis=0) # calcula o novo centroide
                new_centroids.append(cluster_mean) # adiciona a lista de centroides

        # Verifica se houve alguma mudança no centroide
        if np.allclose(centroids, new_centroids): # Retorna True se os dois forem aproximadamente iguais
            increase_cluster = False

        centroids = new_centroids # atualiza os centroides

    return coord_labels, centroids


num_of_clusters = 2
coord_labels, centroids = kmeans(x, y, num_of_clusters)

print("coord_labels:", coord_labels)
print("centroids:", centroids)
