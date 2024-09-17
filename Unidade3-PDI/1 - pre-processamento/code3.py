from src.pdi_utils import load_red_roses, show_image
import matplotlib.pyplot as plt

# Carregar a imagem das rosas vermelhas
image = load_red_roses()

# Mostrar a imagem original
show_image(image, 'Image RGB') y

# Obter o canal vermelho
red_channel = image[:, :, 0]

# Mostrar a imagem do canal vermelho
show_image(red_channel, 'Image Red Channel')

# Plotar o histograma do canal vermelho com 256 bins
plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.7)

# Definir o título e mostrar o histograma
plt.title('Red Histogram')
plt.show()
