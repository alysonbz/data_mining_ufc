# FeatureExtraction.py
import numpy as np
from skimage import color, measure, filters

def calculate_hu_moments(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    moments = measure.moments(img)
    hu_moments = measure.moments_hu(moments)

    return hu_moments

def apply_sobel_edge_detection(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    edges = filters.sobel(img)

    return edges
