import os
import random
from shutil import copyfile

# Diretório contendo as imagens
img_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\classificação de tempo'

# Diretórios para os conjuntos de treinamento e teste
train_dir = "C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\treino"
test_dir = "C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\teste"

# Proporção para dividir entre treinamento e teste (por exemplo, 80% treinamento, 20% teste)
split_ratio = 0.8

# Lista de arquivos no diretório de imagens
all_files = os.listdir(img_dir)

# Embaralhar a lista de arquivos
random.shuffle(all_files)

# Calcular o índice para divisão
split_index = int(len(all_files) * split_ratio)

# Dividir os arquivos em treinamento e teste
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Copiar arquivos para os diretórios de treinamento e teste
for file in train_files:
    src_path = os.path.join(img_dir, file)
    dst_path = os.path.join(train_dir, file)
    copyfile(src_path, dst_path)

for file in test_files:
    src_path = os.path.join(img_dir, file)
    dst_path = os.path.join(test_dir, file)
    copyfile(src_path, dst_path)

# Verificar o número de arquivos nos conjuntos de treinamento e teste
print(f'Número de imagens no conjunto de treinamento: {len(train_files)}')
print(f'Número de imagens no conjunto de teste: {len(test_files)}')
