import os
import random
from shutil import copyfile


img_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\classificação de tempo'


train_dir = "C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\treino"
test_dir = "C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\teste"


split_ratio = 0.8


all_files = os.listdir(img_dir)


random.shuffle(all_files)


split_index = int(len(all_files) * split_ratio)


train_files = all_files[:split_index]
test_files = all_files[split_index:]


for file in train_files:
    src_path = os.path.join(img_dir, file)
    dst_path = os.path.join(train_dir, file)
    copyfile(src_path, dst_path)

for file in test_files:
    src_path = os.path.join(img_dir, file)
    dst_path = os.path.join(test_dir, file)
    copyfile(src_path, dst_path)


print(f'Número de imagens no conjunto de treinamento: {len(train_files)}')
print(f'Número de imagens no conjunto de teste: {len(test_files)}')
