import os
import cv2


def create_dataset(diretory, categories, image_size):
    images = []
    labels = []

    for index, category in enumerate(categories):
        path = os.path.join(diretory, category)

        for image in os.listdir(path):
            image = cv2.imread(os.path.join(path, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, (image_size, image_size)) # padronizar as dimensões
            images.append(resized_image)
            labels.append(index)

    return images, labels

