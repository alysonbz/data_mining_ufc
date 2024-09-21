import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

data = "../documents/features_csv.csv"
df = pd.read_csv(data)

X = df.drop(['label'], axis=1)
y = df['label']

# Treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Normalizando
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

model = GradientBoostingClassifier()

param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 15],
    'learning_rate': [0.01, 0.1],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train_normalized, y_train)

# Melhores parametros
print("Melhores Hiperparâmetros encontrados:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Resultado
y_pred = best_model.predict(X_test_normalized)
print("Accuracy do Melhor Modelo:", accuracy_score(y_test, y_pred))
print("\nClassification Report do Melhor Modelo:\n", classification_report(y_test, y_pred))
