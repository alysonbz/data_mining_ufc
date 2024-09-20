import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv('/data_mining_ufc/AV2/resultados/atributos.csv')

    X = df[['R_mean', 'G_mean', 'B_mean']].values
    y = df['label'].values

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Salvar os conjuntos de treino e teste
    np.save('/data_mining_ufc/AV2/scripts/X_train.npy', X_train)
    np.save('/data_mining_ufc/AV2/scripts/X_test.npy', X_test)
    np.save('/data_mining_ufc/AV2/scripts/y_train.npy', y_train)
    np.save('/data_mining_ufc/AV2/scripts/y_test.npy', y_test)
