import numpy as np


def normalize_images(images):
    normalized_images = []
    for image in images:
        normalized_image = image.astype(np.float32)
        normalized_image = normalized_image/255.0
        normalized_images.append(normalized_image)
    return normalized_images
