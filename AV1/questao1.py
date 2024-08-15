'''Em uma atividade de casa, você implementou manualmente o algoritmo K-means para realizar uma analise não supervisionada.
Nesta avaliação você deve generalizar para receber uma estrutura de mais de duas dimensões e garantir que sua implementação manual realize as seguinte exigências:
a) Deve haver uma classe Kmeans, em que o contrutor ao inicializar, recebe a quantidade de centroides para o algoritmo e o dataframe.
b) O código deve retornar a label que cada ponto do conjunto de dados pertence, com base na quantidade de cluster definida.
c) A definição da quantidade de cluster não pode ser baseada na coluna target do datasset, você deve informar a quantidade de cluster utilizando algum método para encontrar esse parâmetro.'''

#API para importação/dowload de dataset do kaggle
import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset="fedesoriano/heart-failure-prediction", unzip=True)

#importação das bibliotecas e dataset
import pandas as pd
import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, num_of_clusters, max_iterations=100):
        self.num_of_clusters = num_of_clusters
        self.max_iterations = max_iterations

    def fit(self, df):
        self.df = df
        self.coords = self.create_points(df)
        self.centroids, _ = self.set_random_cluster_coordinate()
        self.labels = self.run_kmeans()
        return self.labels

    def create_points(self, df):
        coords = df.values.tolist()  # Aceita qualquer dimensão
        return coords

    def set_random_cluster_coordinate(self):
        coord_list = random.sample(self.coords, self.num_of_clusters)  # Garantindo amostragem única
        label_list = range(self.num_of_clusters)
        return coord_list, label_list

    def dist_euclidean(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def run_kmeans(self):
        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.num_of_clusters)]
            for coord in self.coords:
                distances = [self.dist_euclidean(coord, centroid) for centroid in self.centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(coord)
            
            new_centroids = []
            for cluster in clusters:
                if cluster:  # Verifique se o cluster não está vazio
                    new_centroid = np.mean(cluster, axis=0).tolist()
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(random.choice(self.centroids))  # Escolhe aleatoriamente um dos centróides existentes
            
            if np.all(np.array(new_centroids) == np.array(self.centroids)):
                break
            
            self.centroids = new_centroids
        
        coord_label = [None] * len(self.coords)
        for i, cluster in enumerate(clusters):
            for coord in cluster:
                index = self.coords.index(coord)
                coord_label[index] = i
        
        return coord_label

    def determine_optimal_clusters(self, max_k):
        # Método do cotovelo para determinar o número ideal de clusters
        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(num_of_clusters=k)
            kmeans.fit(self.df)
            distortion = self.calculate_distortion(kmeans.centroids)
            distortions.append(distortion)
        return distortions

    def calculate_distortion(self, centroids):
        distortion = 0
        for coord in self.coords:
            distances = [self.dist_euclidean(coord, centroid) for centroid in centroids]
            distortion += min(distances)
        return distortion

# Exemplo de uso
df = pd.read_csv("heart.csv")


colunas = {}
for column in df.select_dtypes(include=['object']).columns:
    colunas[column] = LabelEncoder()
    df[column] = colunas[column].fit_transform(df[column])

kmeans = KMeans(num_of_clusters=3)
labels = kmeans.fit(df)
print(f'labels: {labels}')

# Encontrar o número ideal de clusters
distortions = kmeans.determine_optimal_clusters(max_k=10)
# Plote os valores de distortions para o método do cotovelo

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), distortions, marker='o', linestyle='--')
plt.title('Método do Cotovelo: Número de Clusters vs Distorção')
plt.xlabel('Número de Clusters')
plt.ylabel('Distorção')
plt.show()

def analyze_clusters(df, labels):
    df['Cluster'] = labels
    cluster_summary = df.groupby('Cluster').mean()  # Média das características por cluster
    return cluster_summary

cluster_summary = analyze_clusters(df, labels)
print(cluster_summary)

