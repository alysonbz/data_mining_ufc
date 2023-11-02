# OBJETIVO -------------------------------------------------------------------------------------------------------------

# Construir uma classe Kmeans que receba uma estrutura de mais de duas dimensões. O algoritmo deve:
#   -> receber o numero de cluster no construtor (OK!)
#   -> retornar a label de cada grupo (OK!)
# PS: A quantidade de clusters deve ser indicada utilizando algum método (OK!)


# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12)


# CLASSE KMEANS --------------------------------------------------------------------------------------------------------

class Kmeans:
    def __init__(self, df, n_clusters):
        self.n_clusters = n_clusters
        self.df = df.to_numpy()
        self.n_samples, self.n_features = df.shape
        self.centroids, self.centroids_labels = self.set_random_cluster_coordinate(n_clusters)

    def set_random_cluster_coordinate(self, n_clusters):
        indices = np.random.choice(self.n_samples, n_clusters, replace=False)
        coords = np.array([self.df[i] for i in indices])
        labels = np.arange(n_clusters)
        return coords, labels

    def dist_euclidean(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def set_closest_centroid(self):
        coord_labels = []
        for coord in self.df:
            distances = [self.dist_euclidean(coord, centroid) for centroid in
                         self.centroids]  # calcula a distancia entre os pontos
            coord_labels.append(np.argmin(distances))  # guarda o indice da menor distancia (label)
        return coord_labels

    def set_new_centroids(self, coord_labels, n_clusters):
        new_centroids = []
        for cluster_i in range(n_clusters):
            cluster_points = [self.df[i] for i, label in enumerate(coord_labels) if
                              label == cluster_i]  # cria uma lista com os pontos atribuidos a cada centroide
            if cluster_points:  # se tiver pontos no cluster atual
                cluster_mean = np.mean(cluster_points, axis=0)  # calcula o novo centroide
                new_centroids.append(cluster_mean)  # adiciona a lista de centroides
        return new_centroids

    def distortion(self, coord_labels):
        distortion = 0
        for cluster_i in range(self.n_clusters):
            cluster_points = [self.df[i] for i, label in enumerate(coord_labels) if label == cluster_i]
            if cluster_points:
                centroid = self.centroids[cluster_i]
                distortion += np.sum((np.array(cluster_points) - centroid) ** 2)
        return distortion

    def kmeans(self):
        increase_cluster = True

        while increase_cluster:
            coord_labels = self.set_closest_centroid()
            new_centroids = self.set_new_centroids(coord_labels, self.n_clusters)

            if np.allclose(self.centroids, new_centroids):
                increase_cluster = False

            self.centroids = new_centroids

        distortion = self.distortion(coord_labels)

        return coord_labels, self.centroids, distortion  # retornar a label de cada grupo


# MÉTODO DE ELBOW ------------------------------------------------------------------------------------------------------
def elbow_method(df, max_lim):
    distortions = []
    num_clusters = range(1, max_lim)

    for i in num_clusters:
        kmeans = Kmeans(df, i)
        _, _, distortion = kmeans.kmeans()
        distortions.append(distortion)

    elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

    sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
    plt.title('Gráfico do método do cotovelo para a escolha do número de clusters')
    plt.xticks(num_clusters)
    plt.show()


# TESTE DO ALGORITMO ---------------------------------------------------------------------------------------------------

# Dados:
df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/breast-cancer.csv')

x = df[['radius_mean', 'texture_mean', 'perimeter_mean']]

# pode também ser utilizado todas as variaveis:
# x = df.drop(['diagnosis', 'id'], axis=1)

# elbow_method(x, 7)

n_clusters = 2  # pela indicação do elbow method

kmeans = Kmeans(x, n_clusters)  # instancia a classe
coord_labels, centroids, _ = kmeans.kmeans()  # chama o metodo

print("coord_labels:", coord_labels)
print("centroids:", centroids)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(x['radius_mean'], x['texture_mean'], x['perimeter_mean'], c=coord_labels, cmap='viridis')
ax.set_title('clusters = 2')
ax.set_ylabel('radius_mean')
ax.set_xlabel('texture_mean')
ax.set_zlabel('perimeter_mean')

plt.show()