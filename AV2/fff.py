# Leitura do Dataset

import os
import pandas as pd
from PIL import Image
import numpy as np

# Caminhos para as pastas
Covid_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Covid'
Normal_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Normal'

# Função para ler imagens e rótulos
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')  # Converte para escala de cinza
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar a imagem {filename}: {e}")
    return images, labels

# Carregar imagens e rótulos
images_covid, labels_covid = load_images_from_folder(Covid_train, 'Covid')
images_normal, labels_normal = load_images_from_folder(Normal_train, 'Normal')

# Combinar dados
images = images_covid + images_normal
labels = labels_covid + labels_normal


# Separação de Treino e Teste

from sklearn.model_selection import train_test_split

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# Extrair Atributos

def extract_features(images):
    features = []
    for img in images:
        mean = np.mean(img)
        std_dev = np.std(img)
        features.append([mean, std_dev])
    return np.array(features)

# Extrair atributos
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)



# Salvar em CSV
df_train = pd.DataFrame(X_train_features, columns=['Mean', 'StdDev'])
df_train['Label'] = y_train

df_test = pd.DataFrame(X_test_features, columns=['Mean', 'StdDev'])
df_test['Label'] = y_test

# Salvar em CSV
df_train.to_csv(r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\train_features.csv', index=False)
df_test.to_csv(r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\test_features.csv', index=False)


# Treinar um Modelo de Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Treinar o modelo
model = LogisticRegression()
model.fit(X_train_features, y_train)

# Fazer previsões
y_pred = model.predict(X_test_features)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Relatório de Classificação:\n{report}")
