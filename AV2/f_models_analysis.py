from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster.vq import whiten
from sklearn.svm import SVC

from AV2.e_attribute_extraction import extract_attributes_met_pixels
from AV2.c_preprocessing import normalize_images
from AV2.b_reading_data import create_dataset


# normalizando pixels --------------------------------------------------------------------------------------------------

diretory_test = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/test'
diretory_train = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/train'

categories = ['default', 'smoke', 'fire']

X_test, y_test = create_dataset(diretory=diretory_test, categories=categories, image_size=200)
X_train, y_train = create_dataset(diretory=diretory_train, categories=categories, image_size=200)


# normalizando pixels --------------------------------------------------------------------------------------------------

X_train = normalize_images(X_train)
X_test = normalize_images(X_test)


# extraindo atributos --------------------------------------------------------------------------------------------------

X_train = extract_attributes_met_pixels(X_train)
X_test = extract_attributes_met_pixels(X_test)


# normalizando ---------------------------------------------------------------------------------------------------------

X_train = whiten(X_train)
X_test = whiten(X_test)


# treinando os modelos -------------------------------------------------------------------------------------------------

def model_training(classifiers, X_train, y_train, X_test, y_test):
    results = []
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((clf_name, accuracy))
    return results.sort(key=lambda x: x[1], reverse=True)


classifiers = [
    ('Árvore de Decisão', DecisionTreeClassifier(random_state=12)),
    ('Random Forest', RandomForestClassifier(random_state=12)),
    ('AdaBoost', AdaBoostClassifier(random_state=12)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=12)),
    ('Stochastic Gradient Boosting (SGB)', SGDClassifier(random_state=12)),
    ('Regressão Logística', LogisticRegression(random_state=12)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=12, max_iter=100)),
    ('MLP', MLPClassifier(random_state=12, max_iter=100))
]

model_training(classifiers, X_train, y_train, X_test, y_test)
