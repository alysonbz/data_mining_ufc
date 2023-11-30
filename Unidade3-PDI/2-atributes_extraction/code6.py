from src.pdi_utils import show_image, load_soaps_image
import numpy as np
import cv2
import matplotlib.pyplot as plt


seed = []

def on_click(event):
    if event.xdata and event.ydata:
        seed.extend([round(event.ydata), round(event.xdata)])
        plt.close()


image = load_soaps_image()

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fig, ax = plt.subplots()
ax.imshow(image)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

seed = tuple(seed)

image = np.array(gray_image)

regiao = np.zeros_like(image)

def diff_pixels(seed, coord):
    return abs(image[seed] - image[coord])

def get_neighbors(pixel, neighborhood, xsup, ysup, visited):
    x, y = pixel
    new_neighbors = [(x, y+1), (x-1, y), (x+1, y), (x, y-1)]
    valid_neighbors = [(nx, ny) for nx, ny in new_neighbors if 0 <= nx < xsup and 0 <= ny < ysup]
    valid_neighbors = [coord for coord in valid_neighbors if diff_pixels(pixel, coord) <= 10 and pixel not in visited]
    neighborhood.extend(valid_neighbors)
    return neighborhood

xsup, ysup = image.shape

neighborhood = [seed]
visited = []

while neighborhood:
    seed = neighborhood.pop()
    neighborhood = get_neighbors(seed, neighborhood, xsup, ysup, visited)
    for coord in neighborhood:
        regiao[coord] = 1
    visited.append(seed)


result_image = gray_image * regiao


plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')


plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Resultado')


plt.show()

