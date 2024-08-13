import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

class Kmeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
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

# Carregando os dados
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Verificar valores nulos
valores_nulos = df.isnull().sum()
print(valores_nulos)
print(df.info())

#Removendo valores nulos
df = df.dropna(subset=["bmi"])

# Mapeando colunas
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == object:
        df[column] = le.fit_transform(df[column])

# Separar os dados da coluna 'id' (se necessário)
data = df.drop('id', axis=1).values

# Encontrar o número ótimo de clusters
find_optimal_clusters(data, max_k=10)

# Ajustar o modelo K-means com o número ótimo de clusters
kmeans = Kmeans(n_clusters=3)
kmeans.fit(data)

# Prever os clusters para cada ponto
labels = kmeans.predict(data)
df['cluster'] = labels

print(df.head())
