from collections import deque
from src.pdi_utils import load_soaps_image
import cv2
import numpy as np
from skimage import segmentation

imagem = None
tolerancia = 10

def on_mouse_click(event, x, y, flags, param):
    global imagem
    if event == cv2.EVENT_LBUTTONDOWN:
        altura, largura, _ = imagem.shape
        if 0 <= x < largura and 0 <= y < altura:
            semente = (y, x)
            crescimento_regiao(semente)

def crescimento_regiao(semente):
    global imagem, tolerancia
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
        vizinhos = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                    (x-1, y+1), (x+1, y+1), (x-1, y-1), (x+1, y-1)]

        for vizinho in vizinhos:
            nx, ny = vizinho
            if 0 <= nx < altura and 0 <= ny < largura and mascara[nx, ny] == 0:
                fila.append((nx, ny))
    objeto_segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara.astype(np.uint8)*255)
    cv2.imshow("Imagem Segmentada", objeto_segmentado)

def ajustar_tolerancia():
    global tolerancia
    nova_tolerancia = input("Digite a nova tolerância (valor inteiro): ")
    try:
        tolerancia = int(nova_tolerancia)
        print(f"Tolerância ajustada para {tolerancia}.")
    except ValueError:
        print("Valor inválido. A tolerância permanecerá a mesma.")
imagem = load_soaps_image()
JANELA_ORIGINAL = "Imagem Original"
TECLA_T = ord('t')
TECLA_ESC = 27
cv2.namedWindow(JANELA_ORIGINAL)
cv2.imshow(JANELA_ORIGINAL, imagem)
cv2.setMouseCallback(JANELA_ORIGINAL, on_mouse_click)
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == TECLA_T:
        ajustar_tolerancia()
    elif key == TECLA_ESC:
        break

cv2.destroyAllWindows()