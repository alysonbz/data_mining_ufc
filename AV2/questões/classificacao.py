from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
dados = r'C:\Users\sheld\Documents\mineracao\data_mining_ufc\AV2\documents\plantas.csv'
df = pd.read_csv(dados)

# Separando X e Y
X = df.drop(labels=['label'], axis=1)
y = df['label']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Definindo o modelo
model = GradientBoostingClassifier()

# Definindo a grade de parâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

# Configurando a busca em grade
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring="accuracy")

# Ajustar o modelo nos dados de treino
grid_search.fit(X_train, y_train)

# Imprimir os melhores parâmetros
print("Melhores Parâmetros:", grid_search.best_params_)

# Usar o melhor modelo para fazer previsões
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Exibir métricas de desempenho
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report do melhor modelo:\n", classification_report(y_test, y_pred))

# Matriz de confusão
matrix = confusion_matrix(y_test, y_pred)

print("\n Matriz de confusão:\n", matrix)

# Criando o heatmap
sns.heatmap(matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
plt.show()