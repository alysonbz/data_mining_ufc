from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from AV2.b_reading_data import create_dataset


# leitura dos dados ----------------------------------------------------------------------------------------------------

diretory_test = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/test'
diretory_train = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/train'

categories = ['default', 'smoke', 'fire']

X_test, y_test = create_dataset(diretory=diretory_test, categories=categories, image_size=200)
X_train, y_train = create_dataset(diretory=diretory_train, categories=categories, image_size=200)


# grid search ----------------------------------------------------------------------------------------------------------

def grid_search(classifier, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, grid_search.best_params_


model_name = ['Regressão Logística', 'MLP', 'SVM', 'AdaBoost', 'Gradient Boosting']

models = [LogisticRegression(),
          MLPClassifier(),
          SVC(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()]

params_grid = [
    # Regressao Logistica
    {'C': [0.001, 0.01, 0.1, 1, 10, 50, 100]},

    # MLP
    {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
     'alpha': [0.0001, 0.001, 0.01, 0.5, 0.1]},

    # SVM
    {'C': [0.001, 0.01, 0.1, 1, 10, 50, 100]},

    # AdaBoost
    {'n_estimators': [50, 100, 150, 200],
     'learning_rate': [0.001, 0.01, 0.1, 0.2]},

    # Gradient Boosting
    {'n_estimators': [50, 100, 150],
     'learning_rate': [0.01, 0.1, 0.2]}
]

for model, name, params in zip(models, model_name, params_grid):
    accuracy, best_params = grid_search(model, params, X_train, y_train, X_test, y_test)
    print(f'{name}\nAcuracy: {accuracy}\nBest params: {best_params}\n')
