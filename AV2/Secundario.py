import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Caminhos para as pastas de treino e teste
Covid_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Covid'
Covid_test = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\test\Covid'
Normal_train = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\train\Normal'
Normal_test = r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\Covid19-dataset\test\Normal'

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

# Carregar imagens de treino e teste para Covid e Normal
images_covid_train, labels_covid_train = load_images_from_folder(Covid_train, 'Covid')
images_normal_train, labels_normal_train = load_images_from_folder(Normal_train, 'Normal')
images_covid_test, labels_covid_test = load_images_from_folder(Covid_test, 'Covid')
images_normal_test, labels_normal_test = load_images_from_folder(Normal_test, 'Normal')

# Combinar dados de treino
X_train = images_covid_train + images_normal_train
y_train = labels_covid_train + labels_normal_train

# Combinar dados de teste
X_test = images_covid_test + images_normal_test
y_test = labels_covid_test + labels_normal_test

# Função para extrair características (média e desvio padrão)
def extract_features(images):
    features = []
    for img in images:
        mean = np.mean(img)
        std_dev = np.std(img)
        features.append([mean, std_dev])
    return np.array(features)

# Extrair atributos de treino e teste
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Armazenar atributos em CSV
df_train = pd.DataFrame(X_train_features, columns=['Mean', 'StdDev'])
df_train['Label'] = y_train
df_train.to_csv(r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\train_features.csv', index=False)

df_test = pd.DataFrame(X_test_features, columns=['Mean', 'StdDev'])
df_test['Label'] = y_test
df_test.to_csv(r'C:\Users\MASTER\OneDrive\Documentos\Matérias\Mineração de dados\data_mining_ufc\projeto_2\projeto_2\test_features.csv', index=False)

# Treinar o modelo de Regressão Logística
model_lr = LogisticRegression()
model_lr.fit(X_train_features, y_train)

# Treinar o modelo de SVM
model_svm = SVC()
model_svm.fit(X_train_features, y_train)

# Treinar o modelo de Árvore de Decisão
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train_features, y_train)

# Fazer previsões no conjunto de teste com Regressão Logística
y_pred_lr = model_lr.predict(X_test_features)

# Fazer previsões no conjunto de teste com SVM
y_pred_svm = model_svm.predict(X_test_features)

# Fazer previsões no conjunto de teste com Árvore de Decisão
y_pred_tree = model_tree.predict(X_test_features)

# Avaliar o modelo de Regressão Logística
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)

# Avaliar o modelo de SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

# Avaliar o modelo de Árvore de Decisão
accuracy_tree = accuracy_score(y_test, y_pred_tree)
report_tree = classification_report(y_test, y_pred_tree)

# Exibir resultados
print(f"Resultados da Regressão Logística:")
print(f"Acurácia: {accuracy_lr}")
print(f"Relatório de Classificação:\n{report_lr}")

print(f"\nResultados do SVM:")
print(f"Acurácia: {accuracy_svm}")
print(f"Relatório de Classificação:\n{report_svm}")

print(f"\nResultados da Árvore de Decisão:")
print(f"Acurácia: {accuracy_tree}")
print(f"Relatório de Classificação:\n{report_tree}")
