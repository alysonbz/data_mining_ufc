import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood
from skimage import color
from src.pdi_utils import load_soaps_image, show_image


# Função para capturar o clique do mouse
def on_click(event, img):
    x = int(event.xdata)
    y = int(event.ydata)

    # Semente do Crescimento de Regiões a partir do clique
    print(f'Coordenadas da semente: ({x}, {y})')
    seed = (y, x)  # Atenção à ordem (linha, coluna)

    # Aplicar a segmentação por crescimento de regiões
    segmented = flood(img, seed, tolerance=0.1)  # Ajuste o valor de tolerance se necessário

    # Criar uma máscara preta com o objeto segmentado
    result = np.zeros_like(img)
    result[segmented] = img[segmented]  # Mantém o objeto na cor original

    # Exibir o resultado
    show_image(result, "Objeto Segmentado")


# Carregar a imagem
image = load_soaps_image()

# Converter para escala de cinza (opcional)
gray_image = color.rgb2gray(image)

# Mostrar a imagem e aguardar clique
fig, ax = plt.subplots()
ax.imshow(image)
cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, image))
plt.show()
