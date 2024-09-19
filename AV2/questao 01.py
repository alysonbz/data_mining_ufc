import os
from PIL import Image  # Biblioteca Pillow para manipulação de imagens
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Obtendo automaticamente os nomes das pastas no diretório atual
base_dir = os.getcwd()  # Pega o diretório atual
folders = {folder: os.path.join(base_dir, folder) for folder in ['water', 'green_area', 'desert', 'cloudy']}

# Função para carregar uma amostra de imagens de cada pasta
def load_sample_images(folder, label, sample_size=500):
    images = []
    labels = []
    all_filenames = os.listdir(folder)
    # Garantir que o tamanho da amostra não exceda o número de imagens disponíveis
    sample_filenames = np.random.choice(all_filenames, size=min(sample_size, len(all_filenames)), replace=False)

    for filename in sample_filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')  # Abrindo a imagem com Pillow
            img = img.resize((128, 128))  # Redimensionando para 128x128
            img_array = np.array(img)  # Convertendo para um array NumPy
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar a imagem {filename}: {e}")

    return np.array(images), np.array(labels)

# Inicializando listas para armazenar as imagens e rótulos
X = []
y = []

# Carregando uma amostra das imagens de cada pasta
for label, folder in folders.items():
    images, labels = load_sample_images(folder, label, sample_size=500)
    X.extend(images)
    y.extend(labels)

# Convertendo as listas para arrays NumPy
X = np.array(X)
y = np.array(y)

# Separação dos dados em conjunto de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Função para extrair atributos (histograma de cores)
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = []
    for i in range(3):  # Loop pelos 3 canais de cor (RGB)
        channel_hist = np.histogram(image[:, :, i], bins=bins[i], range=(0, 256))[0]
        hist.extend(channel_hist)
    hist = np.array(hist).astype('float32')
    hist /= hist.sum()  # Normalizando o histograma
    return hist

# Extraindo atributos das imagens de treino e teste
X_train_features = np.array([extract_color_histogram(img) for img in X_train])
X_test_features = np.array([extract_color_histogram(img) for img in X_test])

# Convertendo os atributos e rótulos para DataFrame e salvando em CSV
train_data = pd.DataFrame(X_train_features)
train_data['label'] = y_train
train_data.to_csv('train_data.csv', index=False)

test_data = pd.DataFrame(X_test_features)
test_data['label'] = y_test
test_data.to_csv('test_data.csv', index=False)

# Treinando o modelo SVM
model = SVC(kernel='linear', class_weight='balanced')  # Ajustando pesos no caso de desbalanceamento
model.fit(X_train_features, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test_features)

# Avaliando o desempenho com métricas detalhadas
print(classification_report(y_test, y_pred))

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')
