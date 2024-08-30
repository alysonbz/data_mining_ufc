# Import the required module
import matplotlib.pyplot as plt
from src.pdi_utils import show_image, load_chest_ray_x
from skimage import exposure

# Carregar a imagem de raio-X
chest_xray_image = load_chest_ray_x()

# Mostrar a imagem original e seu histograma
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of Image')
plt.hist(chest_xray_image.ravel(), bins=256, color='gray')
plt.show()

# Usar equalização de histograma para melhorar o contraste
xray_image_eq = exposure.equalize_hist(chest_xray_image)

# Mostrar a imagem resultante
show_image(xray_image_eq, 'Resulting Image')

# Mostrar o histograma da imagem equalizada
plt.title('Histogram of Equalized Image')
plt.hist(xray_image_eq.ravel(), bins=256, color='gray')
plt.show()
