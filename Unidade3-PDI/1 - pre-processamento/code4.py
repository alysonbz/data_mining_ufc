from src.pdi_utils import show_image , load_chess_image
from skimage.color import rgb2gray
from skimage import color
# Import the otsu threshold function
from skimage.filters import threshold_otsu


chess_pieces_image = load_chess_image()

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = color.rgb2gray(chess_pieces_image)

#show original image
show_image(chess_pieces_image,'Original image')

# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray

# Show the binary image
show_image(binary, 'Binary image')