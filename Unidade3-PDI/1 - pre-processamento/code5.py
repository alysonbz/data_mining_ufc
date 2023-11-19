
from src.pdi_utils import load_page_image, show_image, manual_rgb2gray
from skimage.filters import threshold_otsu, threshold_local

# Load page image and convert it to grayscale manually
page_image = manual_rgb2gray(load_page_image())

# Show the original image
show_image(page_image, 'Global Thresholding')

# Obtain the optimal Otsu global threshold value
global_thresh = threshold_otsu(page_image) #calcula o limiar otimo com metodo de otsu

# Obtain the binary image by applying global thresholding
binary_global = page_image > global_thresh

# Show the binary image obtained using global thresholding
show_image(binary_global, 'Global Thresholding')

# Set the block size to 35 for local thresholding
block_size = 35

# Obtain the optimal local thresholding
local_thresh = threshold_local(page_image, block_size, offset=10)

# Obtain the binary image by applying local thresholding
binary_local = page_image > local_thresh

# Show the binary image obtained using local thresholding
show_image(binary_local, 'Local Thresholding')
