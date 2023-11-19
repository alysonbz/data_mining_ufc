from src.pdi_utils import show_image, load_chess_image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Load chess pieces image
chess_pieces_image = load_chess_image()

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

# Show the original image
show_image(chess_pieces_image, 'Original Image')

# Obtain the optimal threshold value with Otsu
thresh = threshold_otsu(chess_pieces_image_gray) #calcular limiar otimo com metodo otsu

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the binary image
show_image(binary, 'Binary Image')
