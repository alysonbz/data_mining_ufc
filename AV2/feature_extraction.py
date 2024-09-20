import numpy as np

# Constantes
IMAGE_SIZE = (128, 128)
SAMPLE_SIZE = 500
BINS = (8, 8, 8)

def extract_color_histogram(image, bins=BINS):
    hist = []
    for i in range(3):
        channel_hist = np.histogram(image[:, :, i], bins=bins[i], range=(0, 256))[0]
        hist.extend(channel_hist)
    hist = np.array(hist).astype('float32')
    hist /= hist.sum()
    return hist

def extract_features(X):
    return np.array([extract_color_histogram(img) for img in X])