# OBJETIVO -------------------------------------------------------------------------------------------------------------

# 1) O cluster é iniciado selecionando 4 pontos aleatórios (centroides iniciais)
# 2) É calculado a distância de cada ponto aos centroides
# 3) Cada ponto é atribuido ao cluster que tem mais proximidade naquela iteração
# 4) Calcula-se a média de cada cluster e lhe é atribuido como novo centroide
# 5) O processo acaba quando não houver mudança entre centroide do passo atual e do passo anterior


# PACKAGES -------------------------------------------------------------------------------------------------------------
from src.utils import load_pokemon_dataset # dados
import numpy as np                         # manipulação de arrays
import seaborn as sns                      # visualização gráfica
import matplotlib.pyplot as plt


# DADOS ----------------------------------------------------------------------------------------------------------------
x, y = load_pokemon_dataset()
np.random.seed(12)


# FUNÇÕES --------------------------------------------------------------------------------------------------------------

# -> Recebe as listas de coordenadas e o numero de clusters desejado e devolve os centroides e suas respectivas labels
def set_random_cluster_coordinate(x, y, num_of_cluster):
    indices = np.random.choice(len(x), num_of_cluster, replace=False) # define 4 indices aleatorios
    coords = np.array([[x[i], y[i]] for i in indices]) # cria uma lista com os elementos correspondentes
    labels = np.arange(num_of_cluster) # cria uma label para cada grupo
    return coords, labels


# -> Retorna uma lista de coordenadas
def create_points(x, y):
    return np.array(list(zip(x, y)))


# -> Calcula a distancia euclidiana entre dois pontos
def dist_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


# -> Atribui cada pontos ao centróide mais próximo
def set_closest_centroid(coords, centroids):
    coord_labels = []
    for coord in coords:
        distances = [dist_euclidean(coord, centroid) for centroid in centroids] # calcula a distancia entre os pontos
        coord_labels.append(np.argmin(distances)) # guarda o indice da menor distancia (label)
    return coord_labels


# -> Atualiza os centroides com base nos pontos de dados atribuídos
def set_new_centroids(coords, coord_labels, num_of_cluster):
    new_centroids = []
    for cluster_i in range(num_of_cluster):
        cluster_points = [coords[i] for i, label in enumerate(coord_labels) if label == cluster_i] # cria uma lista com os pontos atribuidos a cada centroide
        if cluster_points: # se tiver pontos no cluster atual
            cluster_mean = np.mean(cluster_points, axis=0) # calcula o novo centroide
            new_centroids.append(cluster_mean) # adiciona a lista de centroides
    return new_centroids


def kmeans(x, y, num_of_cluster):

    centroids, centroids_labels = set_random_cluster_coordinate(x, y, num_of_cluster) # centroides iniciais
    coords = create_points(x, y) # coordenadas

    increase_cluster = True

    while increase_cluster:

        # atribui cada coordenada ao centroide mais proximo
        coord_labels = set_closest_centroid(coords, centroids)

        # calcula o centroide novamente
        new_centroids = set_new_centroids(coords, coord_labels, num_of_cluster)

        # Verifica se houve mudanças
        if np.allclose(centroids, new_centroids): # Retorna True se os dois forem aproximadamente iguais
            increase_cluster = False

        centroids = new_centroids # atualiza os centroides

    return coord_labels, centroids


# TESTE ----------------------------------------------------------------------------------------------------------------

num_of_clusters = 2

coord_labels, centroids = kmeans(x, y, num_of_clusters)

print("coord_labels:", coord_labels)
print("centroids:", centroids)


# VISUALIZAÇÃO ---------------------------------------------------------------------------------------------------------

sns.scatterplot(x=x, y=y, hue=coord_labels)
plt.legend(loc='lower right')
plt.show()
