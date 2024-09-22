from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, kernel='linear', class_weight='balanced'):
    model = SVC(kernel=kernel, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def train_adaboost(X_train, y_train, n_estimators=50, random_state=42):
    model = AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia: {accuracy * 100:.2f}%')