# Import Otsu's thresholding method
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from src.pdi_utils import show_image, load_lools_image

# Load the tools image
tools_image = load_lools_image()

# Convert the image to grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal threshold using Otsu's method
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh

# Show the original image
show_image(tools_image, 'original image')

# Show the resulting binary image
show_image(binary_image, 'Binarized image')
