from src.pdi_utils import load_soaps_image
import cv2
import numpy as np
from pynput import mouse

seed = None
image = load_soaps_image()
segmented = None
def region_growing(image, seed, threshold=20):
    h, w = image.shape[:2]
    segmented = np.zeros((h, w), np.uint8)

    # Pilha de crescimento de regiões
    stack = [seed]
    seed_value = image[seed]

    while len(stack) > 0:
        x, y = stack.pop()

        if segmented[x, y] == 0 and abs(int(image[x, y]) - int(seed_value)) < threshold:
            segmented[x, y] = 255  # Marca o pixel como parte da região

            # Verifica os 4 vizinhos
            if x > 0:
                stack.append((x - 1, y))
            if x < h - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < w - 1:
                stack.append((x, y + 1))

    return segmented

def on_click(x, y, button, pressed):
    global seed, segmented

    if pressed and seed is None:  # Captura apenas o primeiro clique
        seed = (y, x)  # Posição do clique como semente (y, x no OpenCV)
        print(f"Seed captured at: {seed}")

        # Aplica o crescimento de regiões a partir da semente
        segmented = region_growing(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), seed)

        # Exibe a imagem original com a região segmentada
        display_segmented_image()

        return False  # Para o listener após o primeiro clique

def display_segmented_image():
    global image, segmented

    result = np.zeros_like(image)
    for i in range(3):  # Para cada canal de cor
        result[:, :, i] = image[:, :, i] * (segmented // 255)

    # Exibe a imagem original e a segmentada
    cv2.imshow("Segmented Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    global image

    # Verifique se a imagem foi carregada
    if image is None:
        print("Erro ao carregar a imagem. Verifique a função load_soaps_image.")
        return

    # Exibe a imagem para captura do clique
    cv2.imshow("Original Image - Click to select seed", image)

    # Listener de clique do mouse
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()  # Fica aguardando o clique do mouse

if __name__ == "__main__":
    main()
