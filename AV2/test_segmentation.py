# reprodução da técnica de segmentação de fogo aplicada no artigo de Jinkyu Ryu & Dongkurl Kwak
# https://doi.org/10.3390/fire5060194

from AV2.b_reading_data import create_dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Leitura dos dados ----------------------------------------------------------------------------------------------------

diretory_test = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/test'
diretory_train = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/train'

categories = ['default', 'smoke', 'fire']

X_test, y_test = create_dataset(diretory=diretory_test, categories=categories, image_size=200)
X_train, y_train = create_dataset(diretory=diretory_train, categories=categories, image_size=200)


# imagens selecionadas -------------------------------------------------------------------------------------------------

default_img = X_train[0]
smoke_img = X_train[345]
fire_img = X_train[489]


# transformando a imagem RGB em HSV ------------------------------------------------------------------------------------

hsv_default_img = cv2.cvtColor(default_img, cv2.COLOR_RGB2HSV)
hsv_smoke_img = cv2.cvtColor(smoke_img, cv2.COLOR_RGB2HSV)
hsv_fire_img = cv2.cvtColor(fire_img, cv2.COLOR_RGB2HSV)


# intervalos de corte para os canais -----------------------------------------------------------------------------------

lower_bound = np.array([5, 40, 220], dtype=np.uint8)
upper_bound = np.array([90, 255, 255], dtype=np.uint8)


# criando e aplicando masks --------------------------------------------------------------------------------------------

mask_default_img = cv2.inRange(hsv_default_img, lower_bound, upper_bound)
mask_smoke_img = cv2.inRange(hsv_smoke_img, lower_bound, upper_bound)
mask_fire_img = cv2.inRange(hsv_fire_img, lower_bound, upper_bound)

seg_default_img = cv2.bitwise_and(default_img, default_img, mask=mask_default_img)
seg_smoke_img = cv2.bitwise_and(smoke_img, smoke_img, mask=mask_smoke_img)
seg_fire_img = cv2.bitwise_and(fire_img, fire_img, mask=mask_fire_img)


# plotando os resultados -----------------------------------------------------------------------------------------------

plt.figure(figsize=(7, 7))

plt.subplot(3, 3, 1)
plt.imshow(default_img)
plt.title('Original', fontsize=10)
plt.ylabel('(a) Default')
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 2)
plt.imshow(hsv_default_img)
plt.title('HSV', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 3)
plt.imshow(seg_default_img)
plt.title('Segmentado', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 4)
plt.imshow(smoke_img)
plt.title('Original', fontsize=10)
plt.ylabel('(b) Smoke')
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 5)
plt.imshow(hsv_smoke_img)
plt.title('HSV', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 6)
plt.imshow(seg_smoke_img)
plt.title('Segmentado', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 7)
plt.imshow(fire_img)
plt.title('Original', fontsize=10)
plt.ylabel('(c) Fire')
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 8)
plt.imshow(hsv_fire_img)
plt.title('HSV', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.subplot(3, 3, 9)
plt.imshow(seg_fire_img)
plt.title('Segmentado', fontsize=10)
plt.yticks([])
plt.xticks([])

plt.tight_layout()
plt.show()


