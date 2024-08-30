import numpy as np
from src.pdi_utils import load_flipped_seville, show_image

flipped_seville = load_flipped_seville()

# Mostrar imagem original
show_image(flipped_seville, 'Seville Flipped')

# Inverter a imagem verticalmente
seville_vertical_flip = np.flipud(flipped_seville)

# Mostrar imagem invertida verticalmente
show_image(seville_vertical_flip, 'Seville Vertical Flipped')

# Inverter a imagem horizontalmente
seville_horizontal_flip = np.fliplr(flipped_seville)

# Mostrar imagem invertida horizontalmente
show_image(seville_horizontal_flip, 'Seville Horizontal Flipped')
