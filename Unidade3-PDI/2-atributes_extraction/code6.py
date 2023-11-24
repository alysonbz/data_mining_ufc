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
    rotulos = segmentation.flood(cinza, semente, tolerance=tolerancia)
    mascara = (rotulos == rotulos[semente])
    mascara = mascara.astype(np.uint8) * 255
    objeto_segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara)
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
JANELA_SEGMENTADA = "Imagem Segmentada"
TECLA_T = ord('t')
cv2.namedWindow(JANELA_ORIGINAL)
cv2.imshow(JANELA_ORIGINAL, imagem)
cv2.setMouseCallback(JANELA_ORIGINAL, on_mouse_click)
TECLA_ESC = 27
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == TECLA_T:
        ajustar_tolerancia()
    elif key == TECLA_ESC:
        break

cv2.destroyAllWindows()
