import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from pre_processing import extract_features
from sklearn.metrics import accuracy_score
from pre_processing import train_images_processed, test_images_processed, test_images, test_labels

def preprocess_data(train_images, test_images, train_labels,test_labels):
    train_features = extract_features(train_images)
    test_features = extract_features(test_images)


    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_normalized = scaler.fit_transform(train_features)
    test_features_normalized = scaler.transform(test_features)

    return train_features_normalized, test_features_normalized, train_labels,test_labels

X_train, X_test, y_train, y_test = preprocess_data(train_images_processed, test_images,  test_images_processed ,test_labels)


# Treina o classificador SVM
clf = SVC(random_state=9)
clf.fit(X_train, y_train)

predictions = clf.predict(y_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(test_labels, predictions)

print(accuracy)