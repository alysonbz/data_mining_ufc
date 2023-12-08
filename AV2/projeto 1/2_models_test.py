import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Definir a função de treinamento de modelos
def model_training(classifiers, X, y, test_size=0.2, random_state=42):
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Padronizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Armazenar os modelos treinados
    trained_models = []

    # Treinar e avaliar cada modelo
    results = []
    for clf_name, clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        trained_models.append((clf_name, clf))
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((clf_name, accuracy))

    # Ordenar os resultados por acurácia em ordem decrescente
    results.sort(key=lambda x: x[1], reverse=True)

    return results, trained_models

# Carregar o DataFrame df_combined_total (substitua 'seu_caminho/df_combined_total.csv' pelo caminho correto)
caminho_arquivo_csv = r'C:\Users\joaod\OneDrive\Documentos\Semestre_2023.2\data_minning_ufc\AV2\projeto 1\df_final\df_combined_total.csv'
df_combined_total = pd.read_csv(caminho_arquivo_csv)

# Embaralhar as linhas do DataFrame
df_combined_total = df_combined_total.sample(frac=1, random_state=42).reset_index(drop=True)

# Substituir 'label' pelo nome real da coluna de labels no seu DataFrame, se for diferente
X = df_combined_total.drop('label', axis=1)  # Remover a coluna de labels para obter as características
y = df_combined_total['label']  # Apenas a coluna de labels

# Definir os classificadores
classifiers = [
    ('Árvore de Decisão', DecisionTreeClassifier(random_state=12)),
    ('Random Forest', RandomForestClassifier(random_state=12)),
    ('AdaBoost', AdaBoostClassifier(random_state=12)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=12)),
    ('Stochastic Gradient Boosting (SGB)', SGDClassifier(random_state=12)),
    ('Regressão Logística', LogisticRegression(random_state=12)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=12, max_iter=10000)),
    ('MLP', MLPClassifier(random_state=12, max_iter=500))
]

# Chamar a função de treinamento de modelos
results, trained_models = model_training(classifiers, X, y)

# Exibir os resultados
print("Resultados:")
print(results)

# ----------------------------------------------------
# Substitua 'label' pelo nome real da coluna de labels no seu DataFrame, se for diferente
X = df_combined_total.drop('label', axis=1)  # Remover a coluna de labels para obter as características
y = df_combined_total['label']  # Apenas a coluna de labels

# Embaralhar as linhas do DataFrame
df_combined_total = df_combined_total.sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

# KNN
knn_model = trained_models[6][1]  # Substitua pelo índice correto do KNN na lista trained_models
y_knn_pred = knn_model.predict(X_test)
plot_confusion_matrix(y_test, y_knn_pred, ['benigno', 'maligno'])

# Random Forest
rf_model = trained_models[1][1]  # Substitua pelo índice correto do Random Forest na lista trained_models
y_rf_pred = rf_model.predict(X_test)
plot_confusion_matrix(y_test, y_rf_pred, ['benigno', 'maligno'])

# SVM
svm_model = trained_models[7][1]  # Substitua pelo índice correto do SVM na lista trained_models
y_svm_pred = svm_model.predict(X_test)
plot_confusion_matrix(y_test, y_svm_pred, ['benigno', 'maligno'])