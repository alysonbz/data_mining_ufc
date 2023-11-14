from src.pdi_utils import show_image
# Import the modules from skimage
from skimage import data, color

# Load the rocket image
rocket = data.rocket()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)

# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')


import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread("C:\\Users\\joaod\\Downloads\\53150885441_5c4f6332f1_o.jpg", cv2.IMREAD_GRAYSCALE)

# Converter a imagem em uma matriz NumPy
matriz_pdi = np.array(imagem)

# Agora, 'matriz_pdi' contém os valores dos pixels da imagem

# Para exibir a matriz
print(matriz_pdi)

import matplotlib.pyplot as plt

# Exibir a imagem
plt.imshow(matriz_pdi, cmap='gray')
plt.show()