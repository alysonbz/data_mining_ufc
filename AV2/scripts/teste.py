import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Carrega o arquivo .npy
X = np.load('/data_mining_ufc/AV2/scripts/X.npy')

# Verifica o formato do array
print(X.shape)
print(X.dtype)

# Exibe a primeira imagem do array
plt.imshow(X[0])
plt.show()

df = pd.read_csv('/data_mining_ufc/AV2/resultados/atributos.csv')