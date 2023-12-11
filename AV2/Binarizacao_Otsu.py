import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.color import rgb2gray
from skimage import io, morphology
from skimage.morphology import skeletonize
from skimage.measure import label as label_image  # Renomeie a função aqui
import os
import numpy as np

def load_images(main_folder, class_names):
    images = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)

        if not os.path.exists(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if os.path.isfile(img_path):
                img = io.imread(img_path)
                images.append(img)
                labels.append(class_name)

    return images, labels

class_names = ['abies_concolor', 'acer_campestre', 'amelanchier_canadensis']

# Carregando imagens e rótulos
images, labels = load_images(r"C:\Users\laura\OneDrive\Área de Trabalho\PLANTAS", class_names)

# Dicionário para controlar quantas imagens foram escolhidas por classe
selected_images = {class_name: False for class_name in class_names}

# Escolha aleatoria de uma imagem de cada classe para a aplicação da limiarização adaptativa, segmentação e esqueletização
np.random.seed(42)
for class_name in class_names:
    class_images = [img for img, label in zip(images, labels) if label == class_name]
    chosen_img = np.random.choice(class_images, replace=False)

    if chosen_img.ndim == 3:
        chosen_img = rgb2gray(chosen_img)

    # limiar adaptativo
    local_thresh = threshold_local(chosen_img, block_size=31, method='mean', offset=0.01)
    binary_img = chosen_img > local_thresh

    # operação de erosão
    eroded_img = morphology.erosion(binary_img)

    # segmentando objetos conectados na imagem erodida
    labeled_image = label_image(eroded_img)

    # esqueletizando a imagem segmentada
    skeleton_image = skeletonize(labeled_image > 0)

    plt.figure(figsize=(20, 5))

    plt.subplot(151)
    plt.imshow(chosen_img, cmap=plt.cm.gray)
    plt.title(f'Imagem Original - Classe: {class_name}')

    plt.subplot(152)
    plt.imshow(binary_img, cmap=plt.cm.gray)
    plt.title('Imagem Binarizada (Limiar Adaptativo)')

    plt.subplot(153)
    plt.imshow(eroded_img, cmap=plt.cm.gray)
    plt.title('Imagem Erodida')

    plt.subplot(154)
    plt.imshow(labeled_image, cmap=plt.cm.nipy_spectral)
    plt.title('Imagem Segmentada')

    plt.subplot(155)
    plt.imshow(skeleton_image, cmap=plt.cm.gray)
    plt.title('Imagem Esqueletizada')

    plt.show()
