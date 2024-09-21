from src.pdi_utils import load_lena
import matplotlib.pyplot as plt

from src.pdi_utils import load_red_roses,show_image
import matplotlib.pyplot as plt

image = load_red_roses()

# Show original image
show_image(image,'image RGB')

# Obtain the red channel
red_channel = image[:, :, 0]
blue_channel = image[:, :, 1]
green_channel = image[:, :, 2]


# Show original image
show_image(red_channel,'image red channel')
show_image(blue_channel,'image blue channel')
show_image(green_channel,'image green channel')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=10)

# Set title and show
plt.title('Blue Histogram')
plt.show()