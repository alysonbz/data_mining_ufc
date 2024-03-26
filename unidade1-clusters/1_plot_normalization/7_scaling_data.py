import numpy as np
from src.utils import load_wine_dataset
from sklearn.preprocessing import StandardScaler
wine = load_wine_dataset()
X = wine.drop(['Quality'],axis=1)

# Create the scaler
scaler = StandardScaler()

X_norm = scaler.fit_transform(X)

print('variancia',np.var(X))

print('variancia do dataset normalizado',np.var(X_norm))
