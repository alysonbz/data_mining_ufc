# Import Gaussian filter
from skimage.filters import gaussian
from src.pdi_utils import load_building_image, show_image

# Carregar a imagem
building_image = load_building_image()

# Aplicar filtro Gaussiano com sigma = 1
gaussian_image_sigma1 = gaussian(building_image, sigma=1, multichannel=True)

# Mostrar a imagem original e a imagem com sigma = 1
show_image(building_image, "Original")
show_image(gaussian_image_sigma1, "Reduced Sharpness Gaussian with Sigma = 1")

# Aplicar filtro Gaussiano com sigma = 5
gaussian_image_sigma5 = gaussian(building_image, sigma=5, multichannel=True)

# Mostrar a imagem com sigma = 5
show_image(gaussian_image_sigma5, "Reduced Sharpness Gaussian with Sigma = 5")

# Aplicar filtro Gaussiano com sigma = 10
gaussian_image_sigma10 = gaussian(building_image, sigma=10, multichannel=True)

# Mostrar a imagem com sigma = 10
show_image(gaussian_image_sigma10, "Reduced Sharpness Gaussian with Sigma = 10")
