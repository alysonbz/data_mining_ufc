# Import Gaussian filter
from skimage.filters import gaussian
from src.pdi_utils import load_building_image, show_image

# Load the building image
building_image = load_building_image()

# Apply Gaussian filter with sigma = 1
gaussian_image_1 = gaussian(building_image, sigma=1, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image_1, "Reduced sharpness Gaussian with sigma = 1")

# Apply Gaussian filter with sigma = 5
gaussian_image_5 = gaussian(building_image, sigma=5, multichannel=True)

# Show resulting image to compare
show_image(gaussian_image_5, "Reduced sharpness Gaussian with sigma = 5")

# Apply Gaussian filter with sigma = 10
gaussian_image_10 = gaussian(building_image, sigma=10, multichannel=True)

# Show resulting image to compare
show_image(gaussian_image_10, "Reduced sharpness Gaussian with sigma = 10")
