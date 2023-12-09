import cv2
import numpy as np
from dataset import train_images, train_labels, test_images, test_labels
import matplotlib.pyplot as plt

def preprocess_images(images):
    processed_images = []
    IMG_SIZE = 300
    for img in images:
        img_array = np.array(img)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        if len(new_array.shape) < 3 or new_array.shape[2] == 1:
            resized_image = cv2.resize(new_array, (IMG_SIZE, IMG_SIZE))
        else:
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))
        processed_images.append(resized_image)
    return processed_images

def plot_images_with_labels(images, labels, num_images):
    num_rows = num_images // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(labels[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

train_images_processed = preprocess_images(train_images)
test_images_processed = preprocess_images(test_images)
