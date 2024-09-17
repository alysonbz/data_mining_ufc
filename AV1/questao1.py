import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # Randomly initialize the centroids by selecting n_clusters random points from the dataset
        np.random.seed(42)
        initial_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_indices]

        for i in range(self.max_iter):
            # Assign labels to each point based on the nearest centroid
            labels = self._assign_labels(X)

            # Calculate new centroids as the mean of the points assigned to each centroid
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence (if the centroids don't change significantly)
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break

            self.centroids = new_centroids

        return self

    def _assign_labels(self, X):
        # Compute the index of the closest centroid for each point
        return pairwise_distances_argmin(X, self.centroids)

    def predict(self, X):
        return self._assign_labels(X)

def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    sse = []
    for k in iters:
        model = Kmeans(n_clusters=k)
        model.fit(data)
        # Calcular a soma dos erros quadrados (SSE)
        distances = np.min(np.linalg.norm(data[:, np.newaxis] - model.centroids, axis=2), axis=1)
        sse.append(np.sum(distances**2))

    plt.plot(iters, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.show()


df = pd.read_csv('diabetes.csv')
data = df.drop('Outcome', axis=1).values  # Supondo que há uma coluna 'target' no dataset que não deve ser usada

# Encontrar o número ideal de clusters usando o método do cotovelo
find_optimal_clusters(data, max_k=10)

# Suponha que a partir do gráfico, determinamos que 3 clusters é o ideal
kmeans = Kmeans(n_clusters=3)
kmeans.fit(data)

# Prever os clusters para cada ponto
labels = kmeans.predict(data)
df['cluster'] = labels

print(df['cluster'].head())
