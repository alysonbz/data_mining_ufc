import os
from data_loading import load_dataset
from data_splitting import split_data
from feature_extraction import extract_features
from csv_saving import save_to_csv
from model_training import train_random_forest, train_knn, train_svm, evaluate_model, train_adaboost


def main():
    base_dir = os.getcwd()

    # 1. Leitura do dataset
    X, y = load_dataset(base_dir)

    # 2. Separação dos dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Extração dos atributos (histograma de cores)
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # 4. Armazenamento dos atributos em CSV
    save_to_csv(X_train_features, y_train, 'train_data.csv')
    save_to_csv(X_test_features, y_test, 'test_data.csv')

    # 5. Treinamento do modelo Random Forest
    print("Avaliação com Random Forest:")
    rf_model = train_random_forest(X_train_features, y_train)
    evaluate_model(rf_model, X_test_features, y_test)

    # 5. Treinamento do modelo KNN
    print("\nAvaliação com KNN:")
    knn_model = train_knn(X_train_features, y_train)
    evaluate_model(knn_model, X_test_features, y_test)

    # 6. Treinamento do modelo SVM
    print("\nAvaliação com SVC:")
    svm_model = train_svm(X_train_features, y_train)
    evaluate_model(svm_model, X_test_features, y_test)

    # 7. Treinamento do modelo Adaboost
    print("\nAvaliação com AdaBoost:")
    ab_model = train_adaboost(X_train_features, y_train)
    evaluate_model(ab_model, X_test_features, y_test)

if __name__ == "__main__":
    main()