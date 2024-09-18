import pandas as pd
import cv2 as cv

# Caminhos corretos para as pastas de treino e teste de Covid e Normal
Covid_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Covid'
Covid_test = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\test\Covid'
Normal_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Normal'
Normal_test = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\test\Normal'

# Verificação de que os caminhos estão corretos
print("Caminho do Covid Train:", Covid_train)
print("Caminho do Covid Test:", Covid_test)
print("Caminho do Normal Train:", Normal_train)
print("Caminho do Normal Test:", Normal_test)



import os
import cv2 as cv

# Verificar arquivos nas pastas de treino e teste
def check_images_in_folder(folder):
    print(f"Arquivos na pasta {folder}:")
    for filename in os.listdir(folder):
        print(filename)

# Testar carregamento de uma imagem
def test_image_loading(folder):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        print(f"Tentando carregar a imagem: {img_path}")
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Falha ao carregar a imagem: {filename}")
        else:
            print(f"Imagem {filename} carregada com sucesso, dimensão: {img.shape}")
            break  # Testar apenas uma imagem

# Verificar arquivos nas pastas
check_images_in_folder(Covid_train)
check_images_in_folder(Normal_train)

# Testar carregamento de uma imagem de cada pasta
test_image_loading(Covid_train)
test_image_loading(Normal_train)


'''import os
import cv2 as cv
import numpy as np

# Função para carregar imagens e associar rótulos
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Lendo em escala de cinza
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Carregar dados de treino (Covid e Normal)
covid_train_images, covid_train_labels = load_images_from_folder(Covid_train, label=1)  # 1 para Covid
normal_train_images, normal_train_labels = load_images_from_folder(Normal_train, label=0)  # 0 para Normal

# Carregar dados de teste (Covid e Normal)
covid_test_images, covid_test_labels = load_images_from_folder(Covid_test, label=1)
normal_test_images, normal_test_labels = load_images_from_folder(Normal_test, label=0)

# Unir os dados de treino e teste
X_train = covid_train_images + normal_train_images
y_train = covid_train_labels + normal_train_labels

X_test = covid_test_images + normal_test_images
y_test = covid_test_labels + normal_test_labels

# Converter para arrays numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Exibir informações sobre os conjuntos
print(f"Número de imagens de treino: {len(X_train)}")
print(f"Número de imagens de teste: {len(X_test)}")
'''