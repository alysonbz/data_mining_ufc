import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(label)
    return images, labels


def generate_dataset(train_folder, test_folder):
    train_images, train_labels = load_images_from_folder(train_folder)
    test_images, test_labels = load_images_from_folder(test_folder)

    return (train_images, train_labels), (test_images, test_labels)

arq_teste = "/home/bbmq/Documentos/mineracao_dados/data_minning_ufc/AV2/Covid19-dataset/test"
arq_treino = "/home/bbmq/Documentos/mineracao_dados/data_minning_ufc/AV2/Covid19-dataset/train"

(train_images, train_labels), (test_images, test_labels) = generate_dataset(arq_treino, arq_teste)