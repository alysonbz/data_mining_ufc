import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, n_clusters, df, max_iters=100):
        self.n_clusters = n_clusters
        self.df = df.astype(float)
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self):
        # Escolhe aleatoriamente os centroides iniciais a partir dos dados
        np.random.seed(42)
        initial_indices = np.random.permutation(self.df.shape[0])[:self.n_clusters]
        self.centroids = self.df.iloc[initial_indices].values

    def assign_clusters(self):
        # Atribui cada ponto ao cluster mais próximo
        distances = np.sqrt(((self.df.values[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def update_centroids(self, labels):
        # Atualiza os centroides com base nas médias dos pontos atribuídos
        new_centroids = np.array([self.df.values[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self):
        self.initialize_centroids()
        for _ in range(self.max_iters):
            labels = self.assign_clusters()
            new_centroids = self.update_centroids(labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return labels


def optimal_number_of_clusters(df, max_k):
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, df=df)
        labels = kmeans.fit()
        inertia = np.sum((df.values - kmeans.centroids[labels]) ** 2)
        inertias.append(inertia)

        if len(np.unique(labels)) > 1:  # Para evitar erro com silhouette_score em apenas um cluster
            silhouette_avg = silhouette_score(df, labels)
        else:
            silhouette_avg = -1
        silhouette_scores.append(silhouette_avg)

    # Plotando o método do cotovelo
    plt.figure(figsize=(14, 6))

    # Gráfico da inércia
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')

    # Gráfico da pontuação de silhueta
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Pontuação de Silhueta')
    plt.title('Pontuação de Silhueta')

    plt.tight_layout()
    plt.show()

    # Determinando o número ótimo de clusters
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k


# Carregando e preparando o dataset
df = pd.read_csv('/home/luissavio/PycharmProjects/data_mining_ufc/AV1/drug200.xls')
df_encoded = pd.get_dummies(df.drop(columns=['Drug']), drop_first=True)

# Determinando o número ótimo de clusters
optimal_k = optimal_number_of_clusters(df_encoded, max_k=10)

# Rodando o KMeans com o número ótimo de clusters
kmeans = KMeans(n_clusters=optimal_k, df=df_encoded)
labels = kmeans.fit()

print(f"Número ótimo de clusters: {optimal_k}")
print(f"Labels dos clusters: {labels}")
