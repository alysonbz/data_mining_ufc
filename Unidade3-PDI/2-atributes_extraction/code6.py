from src.pdi_utils import load_soaps_image, show_image
import matplotlib.pyplot as plt
from pynput import mouse
import numpy as np
from skimage import segmentation


# Carregar a imagem
image = load_soaps_image()

# Variáveis globais para armazenar as coordenadas da semente
seed_x = None
seed_y = None

# Função para lidar com o clique do mouse
def on_click(x, y, button, pressed):
    global seed_x, seed_y
    if pressed:
        seed_x, seed_y = int(x), int(y)
        return False  # Encerrar a captura de eventos após o clique

# Capturar o clique do mouse
with mouse.Listener(on_click=on_click) as listener:
    plt.imshow(image)
    plt.title('Clique para escolher a semente')
    plt.show()
    listener.join()

###PROBLEMA###
# Verificar se as coordenadas da semente estão dentro dos limites da imagem
if seed_x is not None and seed_y is not None and 0 <= seed_x < image.shape[1] and 0 <= seed_y < image.shape[0]:
    # Realizar a segmentação por crescimento de regiões
    seeds = np.zeros_like(image[:, :, 0])
    seeds[seed_y, seed_x] = 1
    segmentation_result = segmentation.flood(image, (seed_y, seed_x), connectivity=1, seed_map=seeds)

    # Criar uma imagem preta e preencher com o objeto segmentado
    segmented_image = np.zeros_like(image)
    segmented_image[segmentation_result] = image[segmentation_result]

    # Exibir a imagem segmentada
    show_image(segmented_image, 'Objeto Segmentado')
else:
    print('Coordenadas da semente não capturadas ou fora dos limites da imagem. Tente novamente.')
