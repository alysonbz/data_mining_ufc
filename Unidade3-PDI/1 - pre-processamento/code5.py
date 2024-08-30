from skimage.color import rgb2gray
from src.pdi_utils import load_page_image, show_image, manual_rgb2gray
from skimage.filters import threshold_otsu, threshold_local

# Carregar e converter a imagem para escala de cinza usando manual_rgb2gray
page_image = manual_rgb2gray(load_page_image())

# Mostrar a imagem original
show_image(page_image, 'Global Thresholding')

# Obter o valor ótimo de limiar global com Otsu
global_thresh = threshold_otsu(page_image)

# Obter a imagem binária aplicando o limiar global
binary_global = page_image > global_thresh

# Mostrar a imagem binária obtida
show_image(binary_global, 'Global Thresholding')

# Definir o tamanho do bloco para limiarização local
block_size = 35

# Obter o limiar local usando o tamanho do bloco e um offset
local_thresh = threshold_local(page_image, block_size, offset=10)

# Obter a imagem binária aplicando a limiarização local
binary_local = page_image > local_thresh

# Mostrar a imagem binária local
show_image(binary_local, 'Local Thresholding')
