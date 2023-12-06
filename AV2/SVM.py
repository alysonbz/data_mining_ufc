from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from criacao_de_rotulos import load_data

if __name__ == "__main__":
    # Diretórios de treino e teste
    train_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\treino'
    test_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\teste'

    # Carrega dados de treino
    X_train, y_train = load_data(train_dir)

    # Carrega dados de teste
    X_test, y_test = load_data(test_dir)

    # Divide o conjunto de treino em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Cria e treina o modelo SVM
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)

    # Avalia o modelo no conjunto de validação
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia no conjunto de validação: {accuracy}")

    # Avalia o modelo no conjunto de teste
    y_test_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no conjunto de teste: {accuracy_test}")



