from src.pdi_utils import show_image, load_chess_image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Carregar a imagem
chess_pieces_image = load_chess_image()

# Converter a imagem para escala de cinza usando rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

# Mostrar a imagem original
show_image(chess_pieces_image, 'Original Image')

# Obter o valor ótimo de limiar com o método de Otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Aplicar o limiar na imagem (binarização)
binary = chess_pieces_image_gray > thresh

# Mostrar a imagem binária
show_image(binary, 'Binary Image')
