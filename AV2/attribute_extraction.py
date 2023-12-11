from scipy.ndimage import sobel
from skimage import measure
import numpy as np
def apply_sobel(images):
    sobel_images = [sobel(img) for img in images]
    return sobel_images

def extract_geometric_features(images):
    geometric_features = []
    for img in images:
        binary_image = img > 0.5
        props = measure.regionprops(measure.label(binary_image))
        area = props[0].area if props else 0
        geometric_features.append(area)
    return np.array(geometric_features)
