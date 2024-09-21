import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss


# Caminhos para as pastas de treino e teste
Covid_train = r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\Covid19-dataset\train\Covid'
Covid_test = r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\Covid19-dataset\test\Covid'
Normal_train = r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\Covid19-dataset\train\Normal'
Normal_test = r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\Covid19-dataset\test\Normal'


# Função para ler imagens e rótulos
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar a imagem {filename}: {e}")
    return images, labels

# Carregar imagens e rótulos
images_covid_train, labels_covid_train = load_images_from_folder(Covid_train, 'Covid')
images_normal_train, labels_normal_train = load_images_from_folder(Normal_train, 'Normal')
images_covid_test, labels_covid_test = load_images_from_folder(Covid_test, 'Covid')
images_normal_test, labels_normal_test = load_images_from_folder(Normal_test, 'Normal')

# Combinar dados
X_train = images_covid_train + images_normal_train
y_train = labels_covid_train + labels_normal_train
X_test = images_covid_test + images_normal_test
y_test = labels_covid_test + labels_normal_test

# Função para extrair características
def extract_features(images):
    features = []
    for img in images:
        mean = np.mean(img)
        std_dev = np.std(img)
        features.append([mean, std_dev])
    return np.array(features)

# Extrair atributos
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Armazenar atributos em CSV
df_train = pd.DataFrame(X_train_features, columns=['Mean', 'StdDev'])
df_train['Label'] = y_train
df_train.to_csv(r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\train_features.csv', index=False)

df_test = pd.DataFrame(X_test_features, columns=['Mean', 'StdDev'])
df_test['Label'] = y_test
df_test.to_csv(r'C:\Users\Amor\PycharmProjects\Mine\data_mining_ufc\projeto 2\projeto 2\test_features.csv', index=False)

# Função para treinar e avaliar modelos
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.decision_function(X_test)) if hasattr(model, 'decision_function') else 'N/A'
    logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'

    print(f"Modelo: {model.__class__.__name__}")
    print(f"Acurácia: {accuracy}")
    print(f"Relatório de Classificação:\n{report}")
    print(f"Matriz de Confusão:\n{confusion}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Log-Loss: {logloss}\n")

# Modelos de classificação
logistic_model = LogisticRegression(max_iter=1000)
svm_model = SVC(probability=True)
tree_model = DecisionTreeClassifier()

# Treinar e avaliar os modelos
train_and_evaluate_model(logistic_model, X_train_features, y_train, X_test_features, y_test)
train_and_evaluate_model(svm_model, X_train_features, y_train, X_test_features, y_test)
train_and_evaluate_model(tree_model, X_train_features, y_train, X_test_features, y_test)