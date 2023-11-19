from src.pdi_utils import load_red_roses, show_image
import matplotlib.pyplot as plt

# Load the red_roses image
image = load_red_roses()

# Show original RGB image
show_image(image, 'Image RGB')

# Obtain the red channel
red_channel = image[:, :, 0]

# Show the red channel image
show_image(red_channel, 'Imagem canal vermelho')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.flatten(), bins=256, color='red', alpha=0.7)

# Set title and show the histogram
plt.title('Histograma Vermelho')
plt.show()
