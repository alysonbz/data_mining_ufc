import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- dataframe --------------------------------------------
def create_dataset_single_label(diretory, label, image_size=128):
    images = []
    labels = []

    for image_name in os.listdir(diretory):
        image_path = os.path.join(diretory, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, (image_size, image_size))
            images.append(resized_image)
            labels.append(label)
        else:
            print(f"Erro: Não foi possível ler a imagem {image_path}.")

    return images, labels


diretorio_saida_benignos = r'C:\\Users\\joaod\\OneDrive\\Documentos\\Semestre_2023.2\\data_minning_ufc\\AV2\\projeto 1\\data_silver\\all_nods_benignos'
diretorio_saida_malignos = r'C:\\Users\\joaod\\OneDrive\\Documentos\\Semestre_2023.2\\data_minning_ufc\\AV2\\projeto 1\\data_silver\\all_nods_malignos'

# Criar dataset para imagens benignas com label 0
images_benignos, labels_benignos = create_dataset_single_label(diretorio_saida_benignos, label=0, image_size=128)

# Criar dataset para imagens malignas com label 1
images_malignos, labels_malignos = create_dataset_single_label(diretorio_saida_malignos, label=1, image_size=128)

# Imprimir informações sobre o conjunto de dados benignos
print("Conjunto de Dados Benignos:")
print("Tamanho do Conjunto de Imagens Benignas:", len(images_benignos))
print("Tamanho do Conjunto de Labels Benignas:", len(labels_benignos))
print("Exemplo de Label Benigna:", labels_benignos[0])

# Visualizar a primeira imagem benigna
plt.imshow(images_benignos[0])
plt.title(f'Label: {labels_benignos[0]}')
plt.show()

# Imprimir informações sobre o conjunto de dados malignos
print("\nConjunto de Dados Malignos:")
print("Tamanho do Conjunto de Imagens Malignas:", len(images_malignos))
print("Tamanho do Conjunto de Labels Malignas:", len(labels_malignos))
print("Exemplo de Label Maligna:", labels_malignos[0])

# Visualizar a primeira imagem maligna
plt.imshow(images_malignos[0])
plt.title(f'Label: {labels_malignos[0]}')
plt.show()

# ----------------------------- extrair atributos  --------------------------------------------
def extract_attributes_hist(images):
    dados = []
    for image in images:
        hist = plt.hist(image.ravel(), bins=16)
        dados.append(hist[0])
    return np.array(dados)


def extract_attributes_pixels(images):
    dados = []
    for image in images:
        flattened_image = image.flatten()
        dados.append(list(flattened_image))
    return np.array(dados)

def extract_attributes_met_pixels(images):
    dados = []
    for image in images:
        dados.append({'mean_pixels': np.mean(image),
                      'std_pixels': np.std(image),
                      'median_pixels': np.median(image)})
    return pd.DataFrame(dados)

def extract_attributes_met_hist(images):
    dados = []
    for image in images:
        hist = plt.hist(image.ravel(), bins=16)
        dados.append({'mean_hist': np.mean(hist[0]),
                      'std_hist': np.std(hist[0]),
                      'median_hist': np.median(hist[0])})
    return pd.DataFrame(dados)




# Extraindo atributos para imagens em tons de cinza
df_met_pixels_benignos = extract_attributes_met_pixels(images_benignos)
df_met_pixels_malignos = extract_attributes_met_pixels(images_malignos)

df_met_hist_benignos = extract_attributes_met_hist(images_benignos)
df_met_hist_malignos = extract_attributes_met_hist(images_malignos)

array_hist_benignos = extract_attributes_hist(images_benignos)
array_hist_malignos = extract_attributes_hist(images_malignos)

array_pixels_benignos = extract_attributes_pixels(images_benignos)
array_pixels_malignos = extract_attributes_pixels(images_malignos)

# Exibir os primeiros registros dos DataFrames
print("DataFrame - Met Píxels Benignos:")
print(df_met_pixels_benignos.head())

print("\nDataFrame - Met Píxels Malignos:")
print(df_met_pixels_malignos.head())

print("\nDataFrame - Met Histograma Benignos:")
print(df_met_hist_benignos.head())

print("\nDataFrame - Met Histograma Malignos:")
print(df_met_hist_malignos.head())

def create_combined_dataset(images, labels, image_size=128):
    # Extrair atributos para imagens em tons de cinza
    df_met_pixels = extract_attributes_met_pixels(images)
    df_met_hist = extract_attributes_met_hist(images)
    array_hist = extract_attributes_hist(images)
    array_pixels = extract_attributes_pixels(images)

    # Adicionar labels aos DataFrames e arrays
    df_met_hist['label'] = labels
    df_combined = pd.concat([df_met_pixels, df_met_hist], axis=1)

    array_hist_with_labels = np.column_stack((array_hist, labels))
    array_pixels_with_labels = np.column_stack((array_pixels, labels))

    # Imprimir os primeiros registros dos DataFrames
    print("\nDataFrame - Met Píxels e Met Histograma Combinados:")
    print(df_combined.head())

    return df_combined, array_hist_with_labels, array_pixels_with_labels

# Suponha que você tenha labels_benignos e labels_malignos
labels_benignos = np.zeros(len(images_benignos))
labels_malignos = np.ones(len(images_malignos))

# Criar conjuntos de dados combinados
df_combined_benignos, array_hist_combined_benignos, array_pixels_combined_benignos = create_combined_dataset(images_benignos, labels_benignos)
df_combined_malignos, array_hist_combined_malignos, array_pixels_combined_malignos = create_combined_dataset(images_malignos, labels_malignos)

# Concatenar os DataFrames de maligno e benigno
df_combined_total = pd.concat([df_combined_benignos, df_combined_malignos], ignore_index=True)

# Exibir os primeiros registros do DataFrame combinado total
print("\nDataFrame Combinado Total:")
print(df_combined_total.head())
print(df_combined_total['label'].value_counts())
print(df_combined_total.columns)

# Especificar o caminho e nome do arquivo CSV
caminho_arquivo_csv = r'C:\Users\joaod\OneDrive\Documentos\Semestre_2023.2\data_minning_ufc\AV2\projeto 1\df_final\df_combined_total.csv'

# Salvar o DataFrame em um arquivo CSV
df_combined_total.to_csv(caminho_arquivo_csv, index=False)

print(f"O DataFrame foi salvo em {caminho_arquivo_csv}.")
