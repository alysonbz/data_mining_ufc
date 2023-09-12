from src.utils import load_pokemon_dataset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

## Anotações
# 1. Especificar um tamanho k de clusters centroides
# 2. Esses k centroides serão inicializados aleatoriamente dentro do espaço
# 3. Cada observação é vinculada temporariamente ao centroide mais próximo (distância euclidiana)
# 4. Então o centroide de cada cluster é calculado novamente com base no grupo formado no passo anterior
# 5. O passo três é executado novamente, o processo continua até que nenhuma obsercação troque de cluster

#### Processo iterativo
### Recalculate cluster centroids
## Reassign observations to nearest  centroid
# Continue until nothing is moving  or being  reassigned anymore

# Dataset
x,y = load_pokemon_dataset()

# Organizando os dados
def create_points(x_cord, y_cord):
    return [[x, y] for x, y in zip(x_cord, y_cord)]


def dist_euclidiana(coordenada,coordenada2):
    return np.sqrt((coordenada[0] - coordenada2[0]) ** 2 + (coordenada[1] - coordenada2[1]) ** 2)


def inicia_centroides(data, n_cluster):
    pontos = np.random.choice(len(data), n_cluster, replace=False)
    centroides = [data[i] for i in pontos]
    return centroides
def clusters(X, centroides):
    # Atribuir cada ponto ao centróide mais próximo
    distancias = np.array([[dist_euclidiana(ponto, centro) for centro in centroides] for ponto in data])
    return np.argmin(distancias,axis=1)

def atualiza_centroide(data,cluster_atual,n_cluster):
    centroides_atualizados = []
    for i in range(n_cluster):
        cluster_pontos = [data[j] for j in range(len(data)) if cluster_atual[j] == i]
        if cluster_pontos == 0:
            centroides_atualizados.append([0,0])
        else:
            media = np.mean(cluster_pontos, axis=0)
            centroides_atualizados.append(media)
    return centroides_atualizados

def kmeans(df, n_cluster):
   centroides = inicia_centroides(df,n_cluster)

   while True:
       cluster_inicia = clusters(df,centroides)

       novos_centroides = atualiza_centroide(df, cluster_inicia, n_cluster)

       if np.all([np.array(centroides[i]) == np.array(novos_centroides[i]) for i in range(n_cluster)]):
           break
       centroides = novos_centroides

   return cluster_inicia,centroides


# vizualizando os dados

plt.scatter(x,y)
# plt.show()

# inicializando os dados
data = create_points(x,y)
n = 2

cluster, centros= kmeans(data,num_of_clusters)

data_final = pd.DataFrame({"X" : [p[0] for p in data], "Y" : [p[1] for p in data], "Cluster" : cluster})

plt.scatter(data_final['X'], data_final['Y'], c=data_final['Cluster'])
plt.show()