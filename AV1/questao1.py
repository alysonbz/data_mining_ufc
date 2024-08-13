'''Em uma atividade de casa, você implementou manualmente o algoritmo K-means para realizar uma analise não supervisionada.
Nesta avaliação você deve generalizar para receber uma estrutura de mais de duas dimensões e garantir que sua implementação manual realize as seguinte exigências:
a) Deve haver uma classe Kmeans, em que o contrutor ao inicializar, recebe a quantidade de centroides para o algoritmo e o dataframe.
b) O código deve retornar a label que cada ponto do conjunto de dados pertence, com base na quantidade de cluster definida.
c) A definição da quantidade de cluster não pode ser baseada na coluna target do datasset, você deve informar a quantidade de cluster utilizando algum método para encontrar esse parâmetro.'''

#API para exportação de dataset
import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset="fedesoriano/heart-failure-prediction", unzip=True)

import pandas as pd
import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin_min

data = pd.read_csv("heart.csv")
print(data.to_string())


class Kmeans:
    def __init__(self, num_of_clusters, df):
        self.num_of_clusters = num_of_clusters
        self.df = df
        self.centroids = None
        self.labels  = None

    def set_random_cluster_coodinate(self):
        coord_list = []
        data = self.df.values
        for _ in range(self.num_of_clusters):
            coord = data[random.randint(0, len(data)-1)]
            coord_list.append(coord)
        self.centroids =np.array(coord)

        

        

