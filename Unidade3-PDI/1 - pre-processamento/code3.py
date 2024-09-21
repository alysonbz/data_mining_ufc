from src.pdi_utils import load_red_roses, show_image
import matplotlib.pyplot as plt

image = load_red_roses()

# Show original image
show_image(image, 'Image RGB')

# Obtain the red channel
red_channel = image[:, :, 0]
blue_channel = image[:, :, 1]
green_channel = image[:, :, 1]

# Show the red channel image
show_image(red_channel, 'Image Red Channel')
show_image(blue_channel, 'Image blue Channel')
show_image(green_channel, 'Image green Channel')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)

# Set title and show
plt.title('Blue Histogram')
plt.show()