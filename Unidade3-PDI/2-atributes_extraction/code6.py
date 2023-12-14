import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from data_mining_ufc.src.pdi_utils import load_soaps_image, load_chess_image


# Função de callback para clique do usuário
def on_click(event):
    if event.xdata and event.ydata:
        seed.extend([round(event.ydata), round(event.xdata)])
        plt.close()


# Função para crescimento de região
def region_growing(image, seed, threshold_factor=0.2):
    xsup, ysup = image.shape
    queue = deque([tuple(seed)])
    visited = set(queue)

    # Valor do pixel na semente
    seed_value = np.int32(image[seed])

    # Limiar dinâmico baseado na média local
    threshold = threshold_factor * np.mean(image)

    while queue:
        pixel = queue.popleft()
        x, y = pixel

        # Coordenadas dos vizinhos
        neighbors = [(x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1)]
        valid_neighbors = [(nx, ny) for nx, ny in neighbors if
                           0 <= nx < xsup and 0 <= ny < ysup and (nx, ny) not in visited]

        for coord in valid_neighbors:
            # Verificar a diferença de intensidade em relação à semente
            if abs(seed_value - np.int32(image[coord])) <= threshold:
                queue.append(coord)
                visited.add(coord)

    return visited


# Carregar imagem de sabonetes
image = load_soaps_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Exibir imagem com interação
seed = []
fig, ax = plt.subplots()
ax.imshow(gray_image, cmap='gray')
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# Converter semente para tupla
seed = tuple(seed)

# Aplicar crescimento de região com ajustes para realçar formatos dos sabonetes
region = region_growing(gray_image, seed, threshold_factor=0.2)

# Criar imagem resultante destacando a região de interesse
result_image = np.zeros_like(gray_image)
for coord in region:
    result_image[coord] = gray_image[coord]

# Exibir imagens (Original e Resultado)
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Resultado - Realce de Formatos dos Sabonetes')

plt.show()