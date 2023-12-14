import os
import numpy as np
import pandas as pd
from skimage import io, color, filters, measure
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


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

def calculate_hu_moments(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    moments = measure.moments(img)
    hu_moments = measure.moments_hu(moments)

    return hu_moments

def apply_sobel_edge_detection(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    edges = filters.sobel(img)

    return edges

def extract_features(images):
    features = [calculate_hu_moments(apply_sobel_edge_detection(img)) for img in images]
    return np.array(features)

def save_features_to_csv(features, labels, output_csv):
    data = np.column_stack((features, labels))
    columns = [f'feature_{i}' for i in range(features.shape[1])] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

def main():
    main_folder = r"C:\Users\laura\OneDrive\Área de Trabalho\PLANTAS"
    class_names = ['abies_concolor', 'acer_campestre', 'amelanchier_canadensis']

    images, labels = load_images(main_folder, class_names)

    # Extração de características
    features = extract_features(images)

    # Normalização de características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Seleção de características
    k_best = SelectKBest(f_classif, k='all')
    features_selected = k_best.fit_transform(features_scaled, labels)

    # Dividir dados para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)

    # Treino do modelo SVM com ajuste de parâmetros
    svm_model = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
    svm_model.fit(X_train, y_train)

    # Validação cruzada
    cv_scores = cross_val_score(svm_model, features_selected, labels, cv=5)
    print(f"Acurácia média na validação cruzada: {np.mean(cv_scores)}")

    # Predição
    y_pred = svm_model.predict(X_test)

    # Avaliação do Modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy}")

if __name__ == "__main__":
    main()
