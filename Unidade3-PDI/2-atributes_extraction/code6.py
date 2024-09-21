import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from skimage import color
from src.pdi_utils import load_soaps_image


def on_mouse_click(event):
    global seed_point
    seed_point = (int(event.ydata), int(event.xdata))
    plt.close()


# Carregar a imagem
image = load_soaps_image()

# Converter a imagem para escala de cinza (se necessário para o crescimento de regiões)
gray_image = color.rgb2gray(image)

# Exibir a imagem e capturar a coordenada do clique do mouse
fig, ax = plt.subplots()
ax.imshow(image)
cid = fig.canvas.mpl_connect('button_press_event', on_mouse_click)
plt.show()

# Crescimento de regiões a partir da semente capturada
if seed_point:
    filled_image = flood_fill(gray_image, seed_point, 1)

    # Criar uma máscara binária a partir da imagem preenchida
    mask = filled_image == 1

    # Aplicar a máscara à imagem original para manter apenas o objeto segmentado
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    # Exibir a imagem segmentada
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.show()

    # Opcionalmente, salvar a imagem resultante
    # plt.imsave('segmented_object.png', segmented_image)

else:
    print("Nenhuma semente foi selecionada.")
