# Import threshold and gray convertor functions
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from src.pdi_utils import show_image, load_lools_image

# Carregar a imagem
tools_image = load_lools_image()

# Converter a imagem para escala de cinza
gray_tools_image = rgb2gray(tools_image)

# Obter o limiar ótimo de Otsu
thresh = threshold_otsu(gray_tools_image)

# Obter a imagem binária aplicando o limiar
binary_image = gray_tools_image > thresh

# Mostrar a imagem original
show_image(tools_image, 'Original Image')

# Mostrar a imagem binarizada
show_image(binary_image, 'Binarized Image')
