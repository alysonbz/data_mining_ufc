from src.pdi_utils import load_lena
import matplotlib.pyplot as plt
from src.pdi_utils import load_red_roses,show_image

# Load the red_roses image
image = load_red_roses()

# Show original RGB image
show_image(image, 'Image RGB')

# Obtain the red channel
red_channel = image[:, :, 0]  # Assuming the red channel is the first channel (index 0)

# Show the red channel image
show_image(red_channel, 'Image Red Channel')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.flatten(), bins=256, color='red', alpha=0.7)

# Set title and show the histogram
plt.title('Red Histogram')
plt.show()