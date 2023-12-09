from src.pdi_utils import load_soaps_image, show_image
import cv2
import numpy as np
from collections import deque
from skimage import segmentation
import matplotlib.pyplot as plt

fig = plt.figure()
seed = []


def dimensoes(img):
    altura, largura, canais = img.shape
    return altura, largura, canais

def onclick(event):
    global seed
    ix, iy = int(round(event.xdata)), int(round(event.ydata))
    seed = (iy, ix)
    plt.close()


def crescimento_regiao(imagem, semente, tolerancia=10):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    mascara = np.zeros_like(cinza)
    fila = deque([semente])
    while fila:
        ponto = fila.popleft()
        x, y = ponto
        if mascara[x, y] != 0:
            continue
        rotulos = segmentation.flood(cinza, ponto, tolerance=tolerancia)
        mascara |= (rotulos == rotulos[semente])
        altura, largura = mascara.shape
        vizinhos = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                    (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]

        for vizinho in vizinhos:
            nx, ny = vizinho
            if 0 <= nx < altura and 0 <= ny < largura and mascara[nx, ny] == 0:
                fila.append((nx, ny))
    objeto_segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara.astype(np.uint8) * 255)
    return objeto_segmentado

def main():
    global seed
    img = load_soaps_image()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    show_image(img)
    img_copia = img.copy()

    a, l, c = dimensoes(img)
    print("Propriedades da imagem:")
    print("Formato:", img.shape)
    print("Nº total de pixels:", img.size)
    print("Data:", img.dtype)
    print(f"Essa é a coordenada inicial:", seed)

    img_com_regiao = crescimento_regiao(img_copia, seed)
    show_image(img_com_regiao)

main()
