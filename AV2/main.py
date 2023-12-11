import os
import numpy as np
import pandas as pd  # Adicione esta linha para trabalhar com DataFrames
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns

from features_extraction import calculate_hu_moments, apply_sobel_edge_detection
from dataload import load_images

def extract_features(images):
    features = [calculate_hu_moments(apply_sobel_edge_detection(img)) for img in images]
    return np.array(features)

def save_features_to_csv(features, labels, output_csv):
    # Combine features and labels into a DataFrame
    data = np.column_stack((features, labels))
    columns = [f'feature_{i}' for i in range(features.shape[1])] + ['label']
    df = pd.DataFrame(data, columns=columns)

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)

def main():
    main_folder = r"C:\Users\laura\OneDrive\Área de Trabalho\PLANTAS"
    class_names = ['abies_concolor', 'acer_campestre', 'amelanchier_canadensis']

    images, labels = load_images(main_folder, class_names)
    features = extract_features(images)

    # Salvar características e rótulos em um arquivo CSV
    output_csv = 'features_geometric.csv'
    save_features_to_csv(features, labels, output_csv)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy}")

if __name__ == "__main__":
    main()
