import cv2
import os
import numpy as np
import pandas as pd
from AV2.utils.extracao_atributos import extracao

# Caminhos para as pastas de imagens das plantas
dataset_paths = {
    'amelanchier_canadensis': r'C:\Documentos\faculdade\5º Sem\Mineração\data_mining_ufc\AV2\imagens\imagens\amelanchier_canadensis',
    'abies_concolor': r'C:\Documentos\faculdade\5º Sem\Mineração\data_mining_ufc\AV2\imagens\imagens\abies_concolor',
    'acer_campestre': r'C:\Documentos\faculdade\5º Sem\Mineração\data_mining_ufc\AV2\imagens\imagens\acer_campestre'
}

X = []
y = []

# Iterar sobre cada classe de plantas
for class_name, class_path in dataset_paths.items():
    if os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            # Leitura da imagem
            image = cv2.imread(image_path)
            if image is not None:
                # Redimensionar a imagem
                image = cv2.resize(image, (64, 64))
                X.append(image)
                y.append(class_name)

# Converter para array
X = np.array(X)
y = np.array(y)

# Extração de atributos
X_features = []
for image in X:
    features = extracao(image)
    X_features.append(features)

# Transformando features em array
X_features = np.array(X_features)

# Armazenando atributos
df = pd.DataFrame(X_features)
df['label'] = y

# Criando CSV
csv_file_path = os.path.join("..", "documents", "plantas.csv")
if not os.path.exists(csv_file_path):
    df.to_csv(csv_file_path, index=False)
    print("CSV Criado")
else:
    print("O arquivo CSV já está na pasta")