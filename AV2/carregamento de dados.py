import os
from PIL import Image
import numpy as np


# Constantes
IMAGE_SIZE = (128, 128)
SAMPLE_SIZE = 500
BINS = (8, 8, 8)

def load_sample_images(folder, label, sample_size=SAMPLE_SIZE):
    images, labels = [], []
    all_filenames = os.listdir(folder)
    sample_filenames = np.random.choice(all_filenames, size=min(sample_size, len(all_filenames)), replace=False)

    for filename in sample_filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar a imagem {filename}: {e}")

    return np.array(images), np.array(labels)


def load_dataset(base_dir):
    folders = {folder: os.path.join(base_dir, folder) for folder in ['water', 'green_area', 'desert', 'cloudy']}
    X, y = [], []
    for label, folder in folders.items():
        images, labels = load_sample_images(folder, label)
        X.extend(images)
        y.extend(labels)
    return np.array(X), np.array(y)