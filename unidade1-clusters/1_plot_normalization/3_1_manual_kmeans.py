#classe sobre kmeans

import random
import numpy as np
from src.utils import load_pokemon_dataset
import pandas as pd
import matplotlib.pyplot as plt

dados_pokemon = load_pokemon_dataset()
valores_x, valores_y = load_pokemon_dataset()

df = pd.DataFrame({'x': valores_x, 'y': valores_y})

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroides = None

    def ajustar_prever(self, X):

        indices_aleatorios = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroides = X[indices_aleatorios]

        for i in range(self.max_iter):
            # atribuir clusters
            grupos_de_clusters = self.atribuir_clusters(X)
            centroides_antigos = self.centroides
            # mover centroides
            self.centroides = self.mover_centroides(X, grupos_de_clusters)
            # verificar conclusão
            if (centroides_antigos == self.centroides).all():
                break

        return grupos_de_clusters

    def atribuir_clusters(self, X):
        grupos_de_clusters = []
        distancias = []

        for linha in X:
            for centroide in self.centroides:
                distancias.append(np.sqrt(np.dot(linha - centroide, linha - centroide)))
            menor_distancia = min(distancias)
            indice_posicao = distancias.index(menor_distancia)
            grupos_de_clusters.append(indice_posicao)
            distancias.clear()

        return np.array(grupos_de_clusters)

    def mover_centroides(self, X, grupos_de_clusters):
        novos_centroides = []

        tipos_de_clusters = np.unique(grupos_de_clusters)

        for tipo in tipos_de_clusters:
            novos_centroides.append(X[grupos_de_clusters == tipo].mean(axis=0))

        return np.array(novos_centroides)
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
X = df[['x', 'y']].values
rotulos_de_clusters = kmeans.ajustar_prever(X)
df['cluster'] = rotulos_de_clusters
dados_agrupados = df[['x', 'y', 'cluster']]
dados_por_cluster = [dados_agrupados[dados_agrupados['cluster'] == i] for i in range(num_clusters)]
cores = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(dados_por_cluster[i]['x'], dados_por_cluster[i]['y'], label=f'Cluster {i}', c=cores[i])
centroides = kmeans.centroides
for i, centroide in enumerate(centroides):
    plt.scatter(centroide[0], centroide[1], marker='X', s=200, c='black', label=f'Centroide {i}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Agrupamento K-Means')
plt.legend()
plt.show()
