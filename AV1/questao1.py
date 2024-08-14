import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, data):
        self.n_clusters = n_clusters
        self.data = data
        self.centroids = None
        self.labels = None

    def initialize_centroids(self):
        """Inicializa os centróides escolhendo aleatoriamente k pontos dos dados."""
        np.random.seed(42)
        initial_centroids_idx = np.random.choice(len(self.data), self.n_clusters, replace=False)
        self.centroids = self.data[initial_centroids_idx]

    def assign_clusters(self):
        """Atribui cada ponto ao cluster mais próximo."""
        self.labels = pairwise_distances_argmin(self.data, self.centroids)

    def update_centroids(self):
        """Atualiza os centróides para a média dos pontos atribuídos a cada cluster."""
        new_centroids = np.array([self.data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        self.centroids = new_centroids

    def fit(self, max_iters=300, tol=1e-4):
        """Executa o algoritmo K-means."""
        self.initialize_centroids()
        for i in range(max_iters):
            previous_centroids = self.centroids.copy()
            self.assign_clusters()
            self.update_centroids()
            if np.all(np.abs(self.centroids - previous_centroids) < tol):
                break

    def predict(self, data):
        """Rotula novos dados com base nos centróides encontrados."""
        return pairwise_distances_argmin(data, self.centroids)

def find_optimal_clusters(data, max_k):
    """Encontra o número ótimo de clusters usando o método do cotovelo."""
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, data=data)
        kmeans.fit()
        sse.append(np.sum((data - kmeans.centroids[kmeans.labels]) ** 2))

    plt.figure(figsize=(6, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de clusters')
    plt.ylabel('Soma dos Quadrados dos Erros (SSE)')
    plt.show()

# Carregar dataset
df = pd.read_csv('breast-cancer.csv')

# 1 para "M" e 0 para "B", maligno e benigno
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


# Escalar os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Encontrar o número ideal de clusters
find_optimal_clusters(scaled_data, max_k=10)

# Definir o número de clusters baseado no gráfico do cotovelo
n_clusters = 3  # Escolha baseada no gráfico

# Instanciar e ajustar o modelo K-means
kmeans = KMeans(n_clusters=n_clusters, data=scaled_data)
kmeans.fit()

# Atribuir rótulos aos dados
df['cluster'] = kmeans.labels

# Contar o número de ocorrências de cada cluster
cluster_counts = df['cluster'].value_counts()

# Exibir o resultado
print(cluster_counts)
