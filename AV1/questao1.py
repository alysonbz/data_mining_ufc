import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt


df = pd.read_csv("bodyfat.csv")
data = df.values

class Kmeans:
    def __init__(self, n_clusters, max_iter=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        np.random.seed(1)
        initial_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_indices]

        for i in range(self.max_iter):
            labels = self._assign_labels(X)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        return pairwise_distances_argmin(X, self.centroids)

    def predict(self, X):
        return self._assign_labels(X)

def find_optimal_clusters(data, max_k):
    sse = []
    for k in range(1, max_k + 1):
        model = Kmeans(n_clusters=k)
        model.fit(data)
        distances = np.min(np.linalg.norm(data[:, np.newaxis] - model.centroids, axis=2), axis=1)
        sse.append(np.sum(distances**2))

    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.title('Método do Cotovelo')
    plt.show()


find_optimal_clusters(data, max_k=10)

kmeans = Kmeans(n_clusters=3)
kmeans.fit(data)

labels = kmeans.predict(data)
df['cluster'] = labels
print(df['cluster'])