from src.utils import load_wine_dataset
from scipy.cluster.vq import whiten
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
wine = load_wine_dataset()
X = wine.drop(['Quality'], axis=1)

# Criar o scaler e ajustar os dados
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Normalização usando a função whiten do scipy
X_whiten = whiten(X)

# Calcular a variância dos dados antes e depois da normalização
variancia_original = np.var(X, axis=0)
variancia_normalizada = np.var(X_norm, axis=0)
variancia_whiten = np.var(X_whiten, axis=0)

print('Variância original:', variancia_original)
print('Variância do dataset normalizado:', variancia_normalizada)
print('Variância do dataset com whiten:', variancia_whiten)
