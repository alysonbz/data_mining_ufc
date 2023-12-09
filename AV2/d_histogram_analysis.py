from AV2.c_preprocessing import normalize_images
from AV2.b_reading_data import create_dataset
from matplotlib import pyplot as plt


# Leitura dos dados ----------------------------------------------------------------------------------------------------

diretory_test = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/test'
diretory_train = 'C:/Users/Thays Ferreira/Documents/projeto 5/data/data/img_data/train'

categories = ['default', 'smoke', 'fire']

X_test, y_test = create_dataset(diretory=diretory_test,
                                categories=categories,
                                image_size=200)
X_train, y_train = create_dataset(diretory=diretory_train,
                                  categories=categories,
                                  image_size=200)


# normalizando imagens -------------------------------------------------------------------------------------------------

#X_train = normalize_images(X_train)
#X_test = normalize_images(X_test)


# transformando imagens ------------------------------------------------------------------------------------------------

# imagens em HSV
#X_train = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in X_train]
#X_test = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in X_test]

# imagens em YCrCb
#X_train = [cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) for image in X_train]
#X_test = [cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) for image in X_test]


# selecionando imagens para o grafico ----------------------------------------------------------------------------------

image_list = [X_train[10], X_train[211], X_train[372], X_train[513], X_train[654]]

fig, axs = plt.subplots(5, 4, figsize=(15, 12))

for i in range(5):
    axs[i, 0].imshow(image_list[i])
    axs[i, 0].axis('off')

    for j, color in enumerate(['Red', 'Green', 'Blue']):
        channel_values = image_list[i][:, :, j].ravel()
        axs[i, j + 1].hist(channel_values, bins=256, alpha=0.7)
        axs[i, j + 1].set_title(f'{color} Channel', fontsize=10)

plt.tight_layout()
plt.show()
