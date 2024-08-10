import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregando o dataset
df = pd.read_csv("waterQuality1.csv")

# Identificando e tratando os valores não numéricos (ex: '#NUM!')
df = df.replace('#NUM!', np.nan)
df = df.dropna()

# Separando os features (exceto is_safe)
X = df.drop('is_safe', axis=1)

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class MyKMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Inicialização aleatória dos centroides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Atribuir cada ponto ao cluster mais próximo
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=1)

            # Recalcular os centroides, verificando clusters vazios
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if cluster_points.shape[0] > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
                #else:
                    # Lidar com clusters vazios (opcional)
                    # Por exemplo, re-inicializar com um ponto aleatório
                    # self.centroids[i] = X[np.random.choice(X.shape[0])]

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=1)

# Find the optimal number of clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Criar um objeto KMeans
kmeans = MyKMeans(n_clusters=3)

# Ajustar o modelo aos dados
kmeans.fit(X_scaled)

# Obter os rótulos dos clusters
labels = kmeans.labels

# Adicionar os rótulos ao DataFrame
df['cluster'] = labels

# Analisar os clusters (opcional)
print(df.groupby('cluster').mean())
