from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from criacao_de_rotulos import load_data

if __name__ == "__main__":

    train_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\treino'
    test_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\teste'


    X_train, y_train = load_data(train_dir)


    X_test, y_test = load_data(test_dir)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)


    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia no conjunto de validação: {accuracy}")


    y_test_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no conjunto de teste: {accuracy_test}")



