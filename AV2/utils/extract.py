from skimage.feature import hog
from skimage.color import rgb2gray


def extract(image):
    gray_image = rgb2gray(image)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features


