import numpy as np
import cv2
def fd_haralick(image):
    image = np.array(image, dtype=np.uint8)

    haralick = mh.features.haralick(image).mean(axis=0)
    return haralick


def fd_histogram(image):
    # compute the color histogram
    image = np.array(image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist.flatten()
    return hist
def extract_features(images):
    features = []
    for image in images:
        hist_feature = fd_histogram(image)
        features.append(global_feature)
    return np.array(features)

*''