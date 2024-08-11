import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregando o dataset
df = pd.read_csv("waterQuality1.csv")

# Identificando e tratando os valores não numéricos (ex: '#NUM!')
df = df.replace('#NUM!', np.nan)

# Converter colunas para tipo numérico e eliminar as linhas com valores faltantes
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Separando os features (exceto 'is_safe')
X = df.drop('is_safe', axis=1)

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Classe personalizada MyKMeans
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
            self.labels = np.argmin(distances, axis=0)  # Corrigido para axis=0

            # Recalcular os centroides, verificando clusters vazios
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if cluster_points.shape[0] > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # Lidar com clusters vazios
                    self.centroids[i] = X[np.random.choice(X.shape[0])]

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)  # Corrigido para axis=0

# Encontrar o número ótimo de clusters (Método do Cotovelo)
wcss = []
for i in range(1, 11):
    kmeans = MyKMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(np.sum(np.min(np.sqrt(((X_scaled - kmeans.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)))

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.grid(True)

# Salvando a figura em vez de mostrar interativamente
plt.savefig('elbow_method.png')

# Criar um objeto MyKMeans com o número de clusters desejado
kmeans = MyKMeans(n_clusters=3)

# Ajustar o modelo aos dados
kmeans.fit(X_scaled)

# Obter os rótulos dos clusters
labels = kmeans.labels

# Adicionar os rótulos ao DataFrame original
df['cluster'] = labels

# Analisar os clusters (opcional)
print(df.groupby('cluster').mean())

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Encontrar o número ótimo de clusters (Método do Cotovelo)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


from sklearn.decomposition import PCA

# Aplicar PCA para reduzir a dimensionalidade a 2 componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ajustar o modelo MyKMeans
kmeans = MyKMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Obter os rótulos dos clusters
labels = kmeans.labels

# Visualizar os clusters em 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Visualização dos Clusters com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()
