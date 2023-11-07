import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Smartwatchprices.csv')
df = pd.DataFrame(data)

# Removendo linhas com NaN e outros valores
df = df.dropna(subset=['Price (USD)'])
df = df[~df['Battery Life (days)'].isin(['Unlimited', 'Not specified'])]  # Remova as linhas com esses valores
df['Battery Life (days)'] = df['Battery Life (days)'].str.replace(' hours', '').str.replace(',', '').astype(float)
df = df[~df['Water Resistance (meters)'].isin(['Not specified', 'NaN'])]  # Remova as linhas com esses valores

# Removendo símbolos e tratando a coluna Price
df['Price (USD)'] = df['Price (USD)'].str.replace('$', '').str.replace(',', '.').astype(float)

# Selecionando apenas as colunas númericas
df = df[['Price (USD)', 'Battery Life (days)', 'Water Resistance (meters)']]

df.to_csv('Smartwatchprices_novo.csv', index=False)

class KMeans:
    def __init__(self, n_clusters, df):
        self.n_clusters = n_clusters
        self.df = df
        self.centroids = self._initialize_centroids()

    def _initialize_centroids(self):     # seleciona aleatoriamente pontos que serão usados como os centróides iniciais para os clusters.
        return self.df.sample(self.n_clusters)

    def _assign_to_clusters(self):
        distances = np.zeros((len(self.df), self.n_clusters))  # matriz criada p/ armazenar as distâncias entre cada ponto de dados e cada um dos centróides
        for i in range(self.n_clusters):
            centroid = self.centroids.iloc[i]
            distances[:, i] = np.linalg.norm(self.df - centroid, axis=1)  # calcula a distância
        return np.argmin(distances, axis=1) # função p/ encontrar o cluster mais próximo p/ cada ponto de dados.

    def _update_centroids(self, labels):
        for i in range(self.n_clusters):
            self.centroids.iloc[i] = self.df[labels == i].mean()   # Atualiza os centróides para ser a média dos pontos atribuídos a cada cluster


    def fit(self, max_iterations=100): # argumento/critério
        for _ in range(max_iterations):
            labels = self._assign_to_clusters() # Chama o método para atribuir pontos aos clusters com base nas posições atuais dos centróides.
            self._update_centroids(labels) # Atualiza os centróides

    def predict(self, new_data):
        # Preveja os clusters para novos dados com base nos centróides atuais
        distances = np.zeros((len(new_data), self.n_clusters))  # matriz p/ armazenar as distâncias entre os novos dados e os centróides dos clusters
        for i in range(self.n_clusters):
            centroid = self.centroids.iloc[i]
            distances[:, i] = np.linalg.norm(new_data - centroid, axis=1) # distância entre cada ponto e o centróide atual
        return np.argmin(distances, axis=1) # argmin p/ encontrar o cluster do centróide mais próximo para cada ponto nos novos dados
def elbow_method(df, max_clusters=10): # cotovelo p/ determinar o número ideal de clusters qnd a distorção começa a diminuir a uma taxa mais lenta.
    distortions = [] # lista para armazenar a distorção para diferentes valores de k.
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(k, df)
        kmeans.fit()
        labels = kmeans.predict(df) #  prever os clusters para os dados
        centroid_distances = np.array([np.linalg.norm(df[labels == i] - kmeans.centroids.iloc[i]) for i in range(k)])
        distortions.append(centroid_distances.sum()) # add na lista
    return distortions # soma das distâncias entre pontos e centróides

if __name__ == "__main__":

    df = pd.read_csv('Smartwatchprices_processed.csv')

    max_clusters = 6

    def elbow_method(df, max_clusters):
        distortions = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(k, df)
            kmeans.fit()
            labels = kmeans.predict(df)
            centroid_distances = np.array([np.linalg.norm(df[labels == i] - kmeans.centroids.iloc[i]) for i in range(k)])
            distortions.append(centroid_distances.sum())
        return distortions

    distortions = elbow_method(df, max_clusters)

    # Gráfico do método do cotovelo
    plt.plot(range(1, len(distortions) + 1), distortions, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Distorção')
    plt.title('Método do Cotovelo')
    plt.show()

    # Número ideal de clusters
    num_clusters = 3

    # Instância class KMeans com o número de clusters
    kmeans = KMeans(num_clusters, df)

    # Ajuste do modelo
    kmeans.fit()

    # Previsões de cluster
    labels = kmeans.predict(df)

    print("Rótulos dos Clusters:", labels)