import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

if __name__ == "__main__":
    # Carregar dados
    X_train = np.load('/data_mining_ufc/AV2/scripts/X_train.npy', allow_pickle=True)
    X_test = np.load('/data_mining_ufc/AV2/scripts/X_test.npy', allow_pickle=True)
    y_train = np.load('/data_mining_ufc/AV2/scripts/y_train.npy', allow_pickle=True)
    y_test = np.load('/data_mining_ufc/AV2/scripts/y_test.npy', allow_pickle=True)

    # Treinar um classificador RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Fazer previsões
    y_pred = clf.predict(X_test)

    # Avaliar o desempenho
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Salvar relatório
    with open('/data_mining_ufc/AV2/resultados/relatorio.txt', 'w') as f:
        f.write(f"Acurácia: {accuracy_score(y_test, y_pred)}\n")
        f.write(classification_report(y_test, y_pred))
