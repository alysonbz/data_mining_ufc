# Import the color module
from skimage.color import rgb2gray

# Import the filters module and sobel function
from skimage.filters import sobel

from src.pdi_utils import load_soaps_image, show_image

# Carregar a imagem
soaps_image = load_soaps_image()

# Converter a imagem para escala de cinza
soaps_image_gray = rgb2gray(soaps_image)

# Aplicar o filtro de detecção de bordas Sobel
edge_sobel = sobel(soaps_image_gray)

# Mostrar a imagem original e a imagem resultante para comparação
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")
