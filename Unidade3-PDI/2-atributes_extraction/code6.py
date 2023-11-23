from src.pdi_utils import load_soaps_image
import cv2
import numpy as np
from skimage import segmentation, color

# Função de callback para o evento de clique do mouse
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (y, x)  # As coordenadas são invertidas para corresponder ao formato (linha, coluna) da matriz
        region_growing(seed)

# Função de crescimento de região
def region_growing(seed):
    # Convertendo a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicando o algoritmo de crescimento de regiões
    labels = segmentation.flood(gray, seed, tolerance=10)

    # Criando uma máscara com o objeto segmentado
    mask = (labels == labels[seed])

    # Convertendo a máscara para o formato uint8
    mask = mask.astype(np.uint8) * 255

    # Extraindo o objeto segmentado
    segmented_object = cv2.bitwise_and(img, img, mask=mask)

    # Exibindo a imagem resultante
    cv2.imshow("Segmented Image", segmented_object)

# Carregando a imagem
img = load_soaps_image()

# Criando uma janela para exibir a imagem
cv2.namedWindow("Original Image")
cv2.imshow("Original Image", img)

# Definindo a função de callback do mouse
cv2.setMouseCallback("Original Image", on_mouse_click)

# Aguardando até que a tecla 'ESC' seja pressionada
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Fechando todas as janelas
cv2.destroyAllWindows()
