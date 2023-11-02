import matplotlib.pyplot as plt
from src.utils import load_comic_con_dataset

# Importe a função de dendrograma
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

df = load_comic_con_dataset()

print(df.columns)


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Carregue o DataFrame e selecione as colunas 'x_scaled' e 'y_scaled'
distance_matrix = linkage(df[['x_scaled', 'y_scaled']])

# Crie o dendrograma
dn = dendrogram(distance_matrix)

# Exiba o dendrograma
plt.show()

