# full_script.py

import os
import csv
import numpy as np
from skimage import io, color, exposure, filters, img_as_ubyte, measure
from skimage.feature import local_binary_pattern, hog
from skimage.morphology import disk, binary_opening, remove_small_objects
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def load_images(main_folder, class_names):
    images = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)

        if not os.path.exists(class_path) or not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if os.path.isfile(img_path):
                img = io.imread(img_path)

                # Normalização
                img_normalized = exposure.rescale_intensity(img)

                images.append(img_normalized)
                labels.append(class_name)

    return images, labels

def cortar_dedo_para_imagens(images, file_names_to_process):
    imagens_cortadas = []

    for img, file_name in zip(images, file_names_to_process):
        if file_name in file_names_to_process:
            # Corta a parte superior da imagem (200 até o final no eixo y)
            img_cortada = img[200:, :]
            imagens_cortadas.append(img_cortada)

    return imagens_cortadas


def otsu_threshold_segmentation_color(img, channel=0):
    # Ajuste do contraste para realçar características
    img = exposure.equalize_adapthist(img)

    if len(img.shape) == 3:
        # Aplica o filtro Sobel ao canal de cor verde
        edges = filters.sobel(img[:, :, channel])

        # limiarização de Otsu nas bordas
        threshold_value = filters.threshold_otsu(edges)
        edges_binary = edges > threshold_value

        # matriz booleana com limiar no canal de cor original
        segmented_img = img[:, :, channel] > 0.4 * filters.threshold_otsu(img[:, :, channel])

        # Combina as informações da limiarização Otsu no canal de cor e nas bordas realçadas pelo Sobel
        combined_img = segmented_img | edges_binary

        # Abertura para remover ruídos
        opened_img = binary_opening(combined_img,
                                    disk(5))

        # Removendo pequenos ruídos após a abertura
        cleaned_img = remove_small_objects(opened_img, min_size=50)  # Ajuste o tamanho mínimo conforme necessário

        return cleaned_img
    else:
        return img

def extract_geometric_features(segmented_img, file_name):
    labeled_img = measure.label(segmented_img, connectivity=2)

    # Calculo das características das regiões conectadas
    props = measure.regionprops(labeled_img)

    # Retornando os atributos geométricos
    geometric_features = []
    for prop in props:
        geometric_features.append({
            "Nome do Arquivo": file_name,
            "Area": prop.area,
            "Perimetro": prop.perimeter,
            "Excentricidade": prop.eccentricity
        })

    return geometric_features

def extract_advanced_features(images):
    features = []

    for img in images:
        gray_img = color.rgb2gray(img)

        # Histograma de Cores
        color_hist = []
        for channel in range(img.shape[2]):
            hist, _ = np.histogram(img[:, :, channel], bins=256, range=(0, 256))
            color_hist.extend(hist)

        # Histograma de Padrões Locais Binários
        lbp_img = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp_img, bins=256, range=(0, 256))

        # Histograma de Orientações de Gradientes
        hog_features, _ = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                              block_norm='L2-Hys', visualize=True)

        # Concatenar todos os descritores
        all_features = np.concatenate((color_hist, lbp_hist, hog_features))

        features.append(all_features)

    return np.array(features)

def save_features_to_csv(features, labels, output_csv):
    data = np.column_stack((features, labels))
    columns = [f'feature_{i}' for i in range(features.shape[1])] + ['label']
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(output_csv, index=False)

def display_sample_images(images, labels, class_names):
    # Cria um dicionário para armazenar uma imagem representativa de cada classe
    sample_images = {class_name: None for class_name in class_names}

    # Encontra a primeira imagem de cada classe
    for img, label in zip(images, labels):
        if sample_images[label] is None:
            # Realiza o pré-processamento na imagem
            segmented_img = otsu_threshold_segmentation_color(img, channel=1)
            segmented_img = exposure.adjust_gamma(segmented_img, gamma=0.5)
            sample_images[label] = segmented_img

            # Se todas as imagens foram encontradas, interrompe a busca
            if all(value is not None for value in sample_images.values()):
                break

    # Exibe as imagens representativas após a limiarização e remoção de ruídos
    for class_name, img in sample_images.items():
        plt.figure()
        plt.imshow(img, cmap='gray')  # Utiliza colormap 'gray' para imagens em tons de cinza
        plt.title(f"Classe: {class_name}")
        plt.show()


def main():
    main_folder = r"C:\Users\laura\OneDrive\Área de Trabalho\PLANTAS"
    class_names = ['abies_concolor', 'acer_campestre', 'amelanchier_canadensis']

    images, labels = load_images(main_folder, class_names)

    # Lista das imagens para corte
    files_to_process = ['13291788381638.jpg', '13291788389752.jpg']

    # Extraindo atributos geométricos
    geometric_features_list = []
    images_to_process = cortar_dedo_para_imagens(images, files_to_process)
    for img, label in zip(images_to_process, labels):
        segmented_img = otsu_threshold_segmentation_color(img, channel=1)
        segmented_img = exposure.adjust_gamma(segmented_img, gamma=0.5)
        geometric_features = extract_geometric_features(segmented_img, label)
        geometric_features_list.extend(geometric_features)

    # arquivo CSV
    csv_filename = "atributos_geometricos.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ["Nome do Arquivo", "Area", "Perimetro", "Excentricidade"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(geometric_features_list)

    # atributos avançados
    advanced_features = extract_advanced_features(images)

    # atributos avançados em CSV
    advanced_csv_filename = 'features_advanced.csv'
    save_features_to_csv(advanced_features, labels, advanced_csv_filename)

    # Dividir dados para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(advanced_features, labels, test_size=0.2, random_state=42)

    # Treino do modelo SVM
    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)

    # Prever no conjunto de teste
    y_pred = svm_model.predict(X_test)

    # precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy}")

    # Exibir uma imagem de cada planta
    display_sample_images(images, labels, class_names)

if __name__ == "__main__":
    main()