# BIBLIOTECAS

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------------


def load_laptop_price():
    df = pd.read_csv("/home/bbmq/Documentos/mineracao_dados/data_minning_ufc/AV1/Dataset/laptopPrice.csv")
    categoricas  = ['ssd', 'os_bit', 'os', 'os_bit',"processor_brand","rating","ram_gb"] # Seleção de 7 atributos para o treinamento
    label_encoders = {}
    for coluna in categoricas:
        le = LabelEncoder()
        label_encoders[coluna] = le
        le.fit(df[coluna])
        df['le_' + coluna] = le.transform(df[coluna])
    return df

df = load_laptop_price()

print("---------------------------------------------------\n")
print(f"FORMATO DOS DADOS:\n {df.shape}\n,{df.head()}\n")
print(f"{df.describe()}")
print("---------------------------------------------------\n")

print(f"DESCRIÇÃO DOS DADOS:\n\n {df.info()}\n")
print("---------------------------------------------------\n")
print(f"EXPLORAÇÃO DOS DADOS:\n ")
print("Checando duplicadas : ")
print(f"{df.duplicated().sum()}")
df = df.drop_duplicates()

print(f"DADOS COM A RETIRADA DAS DUPLICATAS: {df.shape}")
print("---------------------------------------------------\n")

print("LIDANDO COM VARIÁVEIS CATEGÓRICAS:")
print("---------------------------------------------------\n")

print(f" Checando valores valores faltantes{df.isna().sum()}")

# Testando transformar varáveis dummys

df = pd.DataFrame(df,
                  columns=["le_ssd", "le_os", "le_os_bit",
                               "le_processor_brand", "le_rating", "le_os_bit"])

print(f"{df.head()}\n")
print(f"{df.info()}")



print(df.isna().sum())
print(df.head())

print("---------------------------------------------------\n")
print("***************************************************\n")
print("ALGORITIMO K-MEANS")
class Kmeans_manual:
    def __init__(self, data, num_clusters,max_clusters=10):
        self.data = data.to_numpy()
        self.num_clusters = num_clusters
        self.max_clusters = max_clusters

    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def find_optimal_clusters(self):
        wcss = []  # Within-cluster sum of squares
        for i in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
        return wcss

    def determine_optimal_clusters(self):
        wcss = self.find_optimal_clusters()
        plt.plot(range(1, self.max_clusters + 1), wcss)
        plt.title('Método Elbow')
        plt.xlabel('Número de clusters')
        plt.ylabel('WCSS')  #
        plt.show()

        # Método Elbow para encontrar o número ideal de clusters
        return wcss.index(min(wcss)) + 1
    def fit(self):
        print(self.determine_optimal_clusters())
        n_clusters = self.num_clusters
        centroids = self.data[:n_clusters]
        cluster_labels = np.argmin(np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2), axis=1)
        return cluster_labels

# Exemplo de uso:
if __name__ == "__main__":

    df = load_laptop_price()
    df = pd.DataFrame(df,
                      columns=["le_ssd", "le_os", "le_os_bit",
                               "le_processor_brand", "le_rating", "le_os_bit"])

    kmeans_custom = Kmeans_manual(df, num_clusters=2,max_clusters=10)
    cluster_labels = kmeans_custom.fit()
    print(cluster_labels)
    
