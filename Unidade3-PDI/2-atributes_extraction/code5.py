# Import the required module
from skimage import exposure
from src.pdi_utils import show_image, load_aerial_image
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np

def compute_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

# Carregar a imagem aérea
image_aerial = load_aerial_image()

# Calcular e mostrar o histograma da imagem não equalizada
hist_n_eq = np.histogram(image_aerial, bins=256)
plt.title('Histogram of Non-Equalized Image')
plt.plot(hist_n_eq[1][:-1], hist_n_eq[0])
plt.show()

# Realizar a equalização do histograma
image_eq = exposure.equalize_hist(image_aerial)

# Calcular e mostrar o histograma da imagem equalizada
hist_eq = np.histogram(image_eq, bins=256)
plt.title('Histogram of Equalized Image')
plt.plot(hist_eq[1][:-1], hist_eq[0])
plt.show()

# Determinar a média dos valores dos pixels que ocorrem na imagem não equalizada
img_mean_n_eq = np.mean(image_aerial) * 255

# Determinar a média dos valores dos pixels que ocorrem na imagem equalizada
img_mean_eq = np.mean(image_eq) * 255

# Determinar a variância dos valores dos pixels que ocorrem na imagem não equalizada
img_var_n_eq = np.var(image_aerial) * 255**2

# Determinar a variância dos valores dos pixels que ocorrem na imagem equalizada
img_var_eq = np.var(image_eq) * 255**2

# Determinar os pixels que ocorrem com menor frequência na imagem não equalizada
l_freq = 0.2
low_freq_region_n_eq = np.where(hist_n_eq[0] < l_freq * np.max(hist_n_eq[0]))

# Determinar os pixels que ocorrem com menor frequência na imagem equalizada
low_freq_region_eq = np.where(hist_eq[0] < l_freq * np.max(hist_eq[0]))

# Determinar a média dos valores dos pixels que ocorrem com menor frequência na imagem não equalizada
mean_pixel_low_freq_n_eq = np.mean(np.array(low_freq_region_n_eq[0]))

# Determinar a média dos valores dos pixels que ocorrem com menor frequência na imagem equalizada
mean_pixel_low_freq_eq = np.mean(np.array(low_freq_region_eq[0]))

# Determinar a entropia do histograma da imagem não equalizada
hist_entropy_n_eq = compute_entropy(hist_n_eq[0], base=2)

# Determinar a entropia do histograma da imagem equalizada
hist_entropy_eq = compute_entropy(hist_eq[0], base=2)

# Mostrar os valores calculados
print("Média de valores de pixel na imagem não equalizada: ", img_mean_n_eq)
print("Média de valores de pixel na imagem equalizada: ", img_mean_eq)

print("Variância de valores de pixel na imagem não equalizada: ", img_var_n_eq)
print("Variância de valores de pixel na imagem equalizada: ", img_var_eq)

print("Entropia de valores do histograma da imagem não equalizada: ", hist_entropy_n_eq)
print("Entropia de valores do histograma da imagem equalizada: ", hist_entropy_eq)

print("Número de pixels com probabilidade de ocorrência menor que", l_freq, "na imagem não equalizada: ", len(low_freq_region_n_eq[0]))
print("Número de pixels com probabilidade de ocorrência menor que", l_freq, "na imagem equalizada: ", len(low_freq_region_eq[0]))

print("Média de valores de pixel de baixa probabilidade de ocorrência na imagem não equalizada: ", mean_pixel_low_freq_n_eq)
print("Média de valores de pixel de baixa probabilidade de ocorrência na imagem equalizada: ", mean_pixel_low_freq_eq)

# Mostrar a imagem original e a imagem resultante
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting Image')
