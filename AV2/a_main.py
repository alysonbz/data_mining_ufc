from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.cluster.vq import whiten
from sklearn.svm import SVC

from AV2.e_attribute_extraction import extract_attributes_met_pixels
from AV2.c_preprocessing import normalize_images
from AV2.b_reading_data import create_dataset
from AV2.f_models_analysis import model_training


# leitura dos dados ----------------------------------------------------------------------------------------------------

diretory_test = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/test'
diretory_train = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/train'

categories = ['default', 'smoke', 'fire']

X_test, y_test = create_dataset(diretory=diretory_test,
                                categories=categories,
                                image_size=200)
X_train, y_train = create_dataset(diretory=diretory_train,
                                  categories=categories,
                                  image_size=200)


# normalizando pixels --------------------------------------------------------------------------------------------------

X_train = normalize_images(X_train)
X_test = normalize_images(X_test)


# opcao: transformando imagens -----------------------------------------------------------------------------------------

# HSV
# X_train = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in X_train]
# X_test = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in X_test]

# YCrCb
# X_train = [cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) for image in X_train]
# X_test = [cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) for image in X_test]


# extraindo atributos --------------------------------------------------------------------------------------------------

X_train = extract_attributes_met_pixels(X_train)
X_test = extract_attributes_met_pixels(X_test)


# normalizando ---------------------------------------------------------------------------------------------------------

X_train = whiten(X_train)
X_test = whiten(X_test)

# classificando --------------------------------------------------------------------------------------------------------

classifiers = [
    ('AdaBoost', AdaBoostClassifier(random_state=12, learning_rate=0.1, n_estimators=50)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=12, learning_rate=0.2, n_estimators=150)),
    ('Regressão Logística', LogisticRegression(random_state=12, C=10)),
    ('SVM', SVC(random_state=12, C=100)),
    ('MLP', MLPClassifier(random_state=12, max_iter=100, alpha=0.1, hidden_layer_sizes=(50, 50)))
]

model_training(classifiers, X_train, y_train, X_test, y_test)
