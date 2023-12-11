# DataLoader.py
import os
from skimage import io

def load_images(main_folder, class_names):
    images = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)

        if not os.path.exists(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if os.path.isfile(img_path):
                img = io.imread(img_path)
                images.append(img)
                labels.append(class_name)

    return images, labels
