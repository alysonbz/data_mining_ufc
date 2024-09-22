import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from skimage import color, io
from skimage.filters import gaussian
from src.pdi_utils import load_soaps_image

# Função para capturar o ponto de clique do mouse
seed_point = None

def on_click(event):
    """Captura o ponto de clique do mouse na imagem"""
    global seed_point
    # Captura as coordenadas de onde o usuário clicou na imagem
    seed_point = (int(event.ydata), int(event.xdata))
    plt.close()  # Fecha a janela após o clique

# Função para segmentar a imagem usando crescimento de regiões
def segment_image(image, seed_point):
    """Realiza a segmentação por crescimento de regiões a partir de um ponto de semente"""
    # Pré-processamento: converte para escala de cinza
    gray_image = color.rgb2gray(image)

    # Suavização para evitar segmentação irregular (opcional)
    smooth_image = gaussian(gray_image, sigma=1)

    # Aplica o crescimento de regiões
    filled_image = flood_fill(smooth_image, seed_point, 1)

    # Cria uma máscara binária a partir da imagem preenchida
    mask = filled_image == 1

    # Mantém apenas o objeto segmentado na cor original
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    return segmented_image

def main():
    global seed_point

    # Carregar a imagem
    image = load_soaps_image()

    # Exibe a imagem e captura a semente (ponto de clique)
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Clique para escolher o ponto de semente")
    plt.show()

    # Verifica se a semente foi capturada
    if seed_point:
        # Realiza a segmentação
        segmented_image = segment_image(image, seed_point)

        # Exibe a imagem segmentada
        plt.imshow(segmented_image)
        plt.title("Imagem Segmentada")
        plt.show()
    else:
        print("Nenhuma semente foi selecionada.")

if __name__ == "__main__":
    main()
