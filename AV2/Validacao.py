import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from criacao_de_rotulos import load_data


if __name__ == "__main__":
    train_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\treino'
    test_dir = 'C:\\Users\\anime\\PycharmProjects\\data_mining_ufc\\AV2\\teste'

    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)

    # Usar OneVsRestClassifier com o modelo treinado
    classifier = OneVsRestClassifier(model)
    classifier.fit(X_train, y_train)

    y_val_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia no conjunto de validação: {accuracy}")

    y_test_pred = classifier.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no conjunto de teste: {accuracy_test}")

    # Curva ROC
    y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
    n_classes = y_val_bin.shape[1]
    y_val_prob = classifier.decision_function(X_val)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plota a curva ROC para cada classe
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'purple'])  # Adapte as cores conforme suas classes
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC por Classe')
    plt.legend(loc='lower right')
    plt.show()

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    # Visualizar a matriz de confusão com valores
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.title('Matriz de Confusão - Conjunto de Validação')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

    # Relatório de classificação
    print("Relatório de Classificação:")
    print(classification_report(y_val, y_val_pred))



