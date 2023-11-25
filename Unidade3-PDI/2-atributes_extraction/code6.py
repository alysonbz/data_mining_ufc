from src.pdi_utils import load_soaps_image
from src.pdi_utils import show_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cv2 import setMouseCallback

img = load_soaps_image()

fig = plt.figure()

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    print("x = %f, y = %d" % (ix, iy))

    global coords
    coords = [ix,iy]

    plt.close()
    return

"""
def dist_euclidian(coord, coord1):
    np.sqrt((coord[0] - coord1[0]) **2 + (coord[1] - coord1[1]) **2)
"""

def dimensoes(img):
    altura, largura, canais = img.shape
    return altura, largura, canais

# Tentar usar item() de CV2


# Criar uma cópia da imagem(binário) para boder pintar o vizinho.
# Porcentagem (limiar) 0.8
# Analisar se os vizinhos de assemelha e se ele já nao for analizado.
# analisa o vizinho se é semelhante, se for pinta na outra imagem e adiciona em vizinhos.

def vizinhos_calc(obj_img, linha_atual, coluna_atual, limiar, visitados):
    vizinhos = []
    linhas, colunas, _ = obj_img.shape
    valor_pixel_atual = obj_img[int(coluna_atual), int(linha_atual)]

    for i in range(max(0, int(coluna_atual) - 1), min(colunas, int(coluna_atual) + 2)):
        for j in range(max(0, int(linha_atual) - 1), min(linhas, int(linha_atual) + 2)):
            if (i, j) not in visitados:
                valor_pixel = obj_img[i, j]

                if np.all(valor_pixel >= limiar * valor_pixel_atual):
                    vizinhos.append((i, j))
                    visitados.add((i, j))

    return vizinhos, visitados

def crescimento(img, coords, limiar=0.6):
    visitados = set()
    vizinhos_por_visitar = [(coords[1], coords[0])]

    while vizinhos_por_visitar:
        coord_atual = vizinhos_por_visitar.pop(0)
        if coord_atual in visitados:
            continue

        vizinhos, visitados = vizinhos_calc(img, coord_atual[0], coord_atual[1], limiar, visitados)

        if not vizinhos:
            break

        for vizinho in vizinhos:
            img[vizinho[1], vizinho[0]] = [0, 0, 0]

        vizinhos_por_visitar.extend(vizinhos)
        visitados.add(coord_atual)

    black = [0, 0, 0]
    mask = np.all(img != black, axis=-1)
    img[mask] = black

    return img

"""
def crescimento(img, coords, limiar=0.8):
    visitados = set()
    vizinhos_por_visitar = [(coords[0], coords[1])]

    while vizinhos_por_visitar:
        coord_atual = vizinhos_por_visitar.pop(0)
        vizinhos, visitados = vizinhos_calc(img, coord_atual[0], coord_atual[1],visitados ,limiar = 0.8)

        # Marcar os vizinhos na imagem como pretos (ou outra cor desejada)
        for vizinho in vizinhos:
            img[vizinho[0], vizinho[1]] = [0, 0, 0]  # Pixel preto

        vizinhos_por_visitar.extend(vizinhos)

    return img
"""

def main():
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    show_image(img)
    img_copia = img.copy()

    a,l,c = dimensoes(img)
    print("Propriedades da imagem:")
    print("Formato:" + str(img.shape))
    print("Nº total de pixels:" + str(img.size))
    print("Data :" + str(img.dtype))
    print(f"Essa é a coordenada inicial : ", coords)
    #vizinhos_vetor = vizinhos(img, coords[0], coords[1])
    img_com_regiao = crescimento(img_copia, coords)

    show_image(img_com_regiao)
main()