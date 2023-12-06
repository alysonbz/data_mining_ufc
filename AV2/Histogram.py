import cv2

def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    else:
        print(f"Erro ao ler a imagem: {image_path}")
        return None
