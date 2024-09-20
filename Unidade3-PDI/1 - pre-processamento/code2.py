import numpy as np
from src.pdi_utils import load_flipped_seville, show_image

# Load the flipped Seville image
flipped_seville = load_flipped_seville()

# Show the original flipped image
show_image(flipped_seville, 'Seville Flipped')

# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Show the image flipped vertically
show_image(seville_vertical_flip, 'Seville Vertical Flipped')

# Flip the image horizontally
seville_horizontal_flip = np.fliplr(flipped_seville)

# Show the image flipped horizontally
show_image(seville_horizontal_flip, 'Seville Horizontal Flipped')