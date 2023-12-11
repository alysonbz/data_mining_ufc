import imgaug.augmenters as iaa
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(images):
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10)),
        iaa.GaussianBlur(sigma=(0, 1.0))
    ])

    augmented_images = augmenter(images=[np.array(img) for img in images])

    return [Image.fromarray(img) for img in augmented_images]

def augment_train_test_data(X_train, X_test, y_train, y_test):
    # Aplica aumento de dados apenas nas imagens de treino
    augmented_train_images = augment_data(X_train)

    # Adiciona imagens aumentadas aos dados originais
    X_train_augmented = X_train + augmented_train_images
    y_train_augmented = y_train + y_train * len(augmented_train_images)

    # Aplica aumento de dados nas imagens de teste
    augmented_test_images = augment_data(X_test)

    # Adiciona imagens aumentadas aos dados originais
    X_test_augmented = X_test + augmented_test_images
    y_test_augmented = y_test + y_test * len(augmented_test_images)

    return X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented

if __name__ == "__main__":
    # Supondo que você já tenha os conjuntos X_train, X_test, y_train, y_test
    # ...

    # Aplica aumento de dados nos conjuntos de treino e teste
    X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented = augment_train_test_data(X_train, X_test, y_train, y_test)

    # Exemplo: exibe a primeira imagem de treino original e aumentada
    original_train_image = X_train[0]
    augmented_train_image = X_train_augmented[0]

    original_train_image.show(title="Original Training Image")
    augmented_train_image.show(title="Augmented Training Image")
