from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processing import train_images_processed, train_labels, test_images_processed, test_labels
import polars as pl
import numpy as np


def normalize_data(train_images, test_images):
    train_images = np.array(train_images_processed)
    test_images = np.array(test_images_processed)
    train_images_flatten = train_images.reshape(train_images.shape[0], -1)
    test_images_flatten = test_images.reshape(test_images.shape[0], -1)

    scaler = StandardScaler()
    scaler.fit(train_images_flatten)
    train_images_normalized = scaler.transform(train_images_flatten)
    test_images_normalized = scaler.transform(test_images_flatten)

    return train_images_normalized, test_images_normalized

def train_svm_model(train_images, train_labels):
    svm_model = svm.SVC(kernel='linear', gamma='auto')
    svm_model.fit(train_images, train_labels)
    return svm_model

def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)

    return accuracy, report, conf_matrix

train_images_normalized, test_images_normalized = normalize_data(train_images_processed, test_images_processed)

svm_model = train_svm_model(train_images_normalized, train_labels)

accuracy, report, conf_matrix = evaluate_model(svm_model, test_images_normalized, test_labels)

print(f"Acurácia do modelo SVM: {accuracy}")
print(pl.DataFrame({'original': test_labels, 'predicted': svm_model.predict(test_images_normalized)}))
print("Relatório de Classificação:")
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
