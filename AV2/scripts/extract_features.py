import numpy as np
import pandas as pd


# Função para extrair a média dos canais RGB de cada imagem
def extract_features(X):
    features = []
    for img in X:
        r_mean = np.mean(img[:, :, 0])  # Canal Red
        g_mean = np.mean(img[:, :, 1])  # Canal Green
        b_mean = np.mean(img[:, :, 2])  # Canal Blue
        features.append([r_mean, g_mean, b_mean])
    return np.array(features)


if __name__ == "__main__":
    X = np.load('/data_mining_ufc/AV2/scripts/X.npy')
    y = np.load('/data_mining_ufc/AV2/scripts/y.npy')

    # Extrair atributos
    features = extract_features(X)

    # Criar DataFrame
    df = pd.DataFrame(features, columns=['R_mean', 'G_mean', 'B_mean'])
    df['label'] = y

    # Salvar como CSV
    df.to_csv('/data_mining_ufc/AV2/resultados/atributos.csv', index=False)
