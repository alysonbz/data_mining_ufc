from src.pdi_utils import load_red_roses, show_image
import matplotlib.pyplot as plt

# Carregar a imagem
image = load_red_roses()

# Mostrar imagem original
show_image(image, 'Image RGB')

# Obter o canal vermelho da imagem
red_channel = image[:, :, 0]

# Mostrar o canal vermelho da imagem
show_image(red_channel, 'Image Red Channel')

# Plotar o histograma do canal vermelho com bins na faixa de 256
plt.hist(red_channel.ravel(), bins=256, color='red')

# Definir título e exibir o gráfico
plt.title('Red Histogram')
plt.show()
