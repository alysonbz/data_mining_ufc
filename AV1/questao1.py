'''' Instruções:

Em uma atividade de casa, você implementou manualmente o algoritmo K-means para realizar uma analise não supervisionada.
Nesta avaliação você deve generalizar para receber uma estrutura de mais de duas dimensões e garantir que sua implementação manual realize as seguinte exigências:
a) Deve haver uma classe Kmeans, em que o construtor ao inicializar, recebe a quantidade de centroides para o algoritmo e o dataframe.
b) O código deve retornar a label que cada ponto do conjunto de dados pertence,quantidade de centroides para o algoritmo e o dataframe.
b) O código deve retornar a label que cada ponto do conjunto de dados pertence, com base na quantidade de cluster definida.
c) A definição da quantidade de cluster não pode ser baseada na coluna target do dataset, você deve informar a quantidade de cluster utilizando algum método para encontrar esse parâmetro.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd

df = pd.read_csv(r"C:\Users\Guilherme\Downloads\Nova Pasta\laptopPrice.csv")

# Definiçaõ da classe KMeansManual
class KMeansManual:
    def __init__(self, n_clusters=3, df=None, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.df = df
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self):
        np.random.seed(self.random_state)
        self.centroides = self.df[np.random.choice(self.df.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            self.labels_ = pairwise_distances_argmin(self.df, self.centroides)
            novos_centroides = np.array([self.df[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroides == novos_centroides):
                break
            self.centroides = novos_centroides

        self.inercia_ = np.sum([np.linalg.norm(self.df[self.labels_ == i] - self.centroides[i], axis=1).sum() for i in range(self.n_clusters)])
        return self.labels_, self.n_clusters, self.df

    def elbow_method(self, k_max=10):
        inercia = []
        for k in range(1, k_max + 1):
            kmeans = KMeansManual(n_clusters=k, df=self.df, max_iter=self.max_iter, random_state=self.random_state)
            kmeans.fit()
            inercia.append(kmeans.inercia_)
        plt.plot(range(1, k_max + 1), inercia, 'o-', color='hotpink')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo (Elbow Method)')
        plt.show()

# Função de pré-processamento
def preprocess_data(df):
    return StandardScaler().fit_transform(df.select_dtypes(include=[int, float]))

# Aplicação do KMeansManual
df_scaled = preprocess_data(df)

kmeans_manual = KMeansManual(df=df_scaled)
kmeans_manual.elbow_method()

kmeans_manual = KMeansManual(n_clusters=4, df=df_scaled)
labels, n_clusters, df_final = kmeans_manual.fit()
df['Cluster'] = labels

print(df.head(10))

plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=labels, cmap='cool')
plt.scatter(kmeans_manual.centroides[:, 0], kmeans_manual.centroides[:, 1], s=300, c='black')
plt.title(f'Clusterização KMeans com 4 clusters')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()