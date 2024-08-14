import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Inicializa os centróides aleatoriamente
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Calcula as distâncias entre cada ponto e os centróides
            distances = cdist(X, self.centroids, 'euclidean')
            # Atribui rótulos baseados na menor distância
            self.labels = np.argmin(distances, axis=1)
            # Calcula novos centróides
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            # Verifica se os centróides convergiram
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)

def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sse.append(np.sum(np.min(cdist(data, kmeans.centroids, 'euclidean'), axis=1)))

    plt.figure(figsize=(10, 8))
    plt.plot(iters, sse, '-o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Erros Quadráticos')
    plt.title('Método do Cotovelo')
    plt.show()

# Exemplo de uso
df = pd.read_csv('C:\\Users\\Amor\\PycharmProjects\\Mine\\data_mining_ufc\\AV1\\df1_pp')

# Encontre o número ideal de clusters
find_optimal_clusters(df, max_k=10)

# Supondo que você escolheu 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
labels = kmeans.predict(df)
print("Rótulos dos clusters: ", labels)
