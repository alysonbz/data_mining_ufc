import matplotlib.pyplot as plt
from src.utils import load_comic_con_dataset

# Importe a função de dendrograma
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

df = load_comic_con_dataset()

# Calcula a matriz de distância
distance_matrix = linkage(df[['x', 'y']])

# Crie um dendrograma
dn = dendrogram(distance_matrix)

# Exiba o dendrograma
plt.show()
