import os
from PIL import Image
import numpy as np
import pandas as pd


# Função para ler imagens e atribuir labels
# Função para ler imagens e atribuir labels
def load_dataset(image_dir):
    images = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert('RGB')  # Converte para RGB
            img = img.resize((64, 64))  # Redimensiona para acelerar o processamento
            img_array = np.array(img)

            # Atribui o rótulo com base no nome do arquivo
            if 'cloudy' in filename:
                labels.append('cloudy')
            elif 'rain' in filename:
                labels.append('rain')
            elif 'shine' in filename:
                labels.append('shine')
            elif 'sunrise' in filename:
                labels.append('sunrise')

            images.append(img_array)

    # Converte para arrays NumPy
    X = np.array(images)
    y = np.array(labels)

    return X, y


if __name__ == "__main__":
    X, y = load_dataset('/data_mining_ufc/AV2/classificacao_de_tempo')
    # Salva para uso nos próximos scripts
    np.save('/data_mining_ufc/AV2/scripts/X.npy', X)
    np.save('/data_mining_ufc/AV2/scripts/y.npy', y)
