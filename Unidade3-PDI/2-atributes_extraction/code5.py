from skimage import exposure
from src.pdi_utils import show_image, load_aerial_image
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np

def calcular_entropia(labels, base=None):
    valores, contagens = np.unique(labels, return_counts=True)
    return entropy(contagens, base=base)
imagem_aerea = load_aerial_image()

hist_n_eq, bin_edges_n_eq = np.histogram(imagem_aerea, bins=256)
plt.plot(bin_edges_n_eq[0:-1], hist_n_eq)
plt.title('Histograma da Imagem Não Equalizada')
plt.show()

imagem_eq = exposure.equalize_hist(imagem_aerea)

hist_eq, bin_edges_eq = np.histogram(imagem_eq, bins=256)
plt.plot(bin_edges_eq[0:-1], hist_eq)
plt.title('Histograma da Imagem Equalizada')
plt.show()

media_pixel_n_eq = np.mean(imagem_aerea) * 255
media_pixel_eq = np.mean(imagem_eq) * 255

variancia_pixel_n_eq = np.var(imagem_aerea) * 255
variancia_pixel_eq = np.var(imagem_eq) * 255

freq_limite = 0.2
regiao_baixa_freq_n_eq = np.where(hist_n_eq < freq_limite * np.max(hist_n_eq))

regiao_baixa_freq_eq = np.where(hist_eq < freq_limite * np.max(hist_eq))
media_pixel_baixa_freq_n_eq = np.mean(regiao_baixa_freq_n_eq[0])
media_pixel_baixa_freq_eq = np.mean(regiao_baixa_freq_eq[0])

entropia_hist_n_eq = calcular_entropia(hist_n_eq, base=2)
entropia_hist_eq = calcular_entropia(hist_eq, base=2)
show_image(imagem_aerea, 'Original')
show_image(imagem_eq, 'Imagem Resultante')
print("Valores médios dos pixels na imagem não equalizada: ", media_pixel_n_eq)
print("Valores médios dos pixels na imagem equalizada: ", media_pixel_eq)

print("Variâncias dos valores dos pixels na imagem não equalizada: ", variancia_pixel_n_eq)
print("Variâncias dos valores dos pixels na imagem equalizada: ", variancia_pixel_eq)

print("Entropia dos valores dos pixels na imagem não equalizada: ", entropia_hist_n_eq)
print("Entropia dos valores dos pixels na imagem equalizada: ", entropia_hist_eq)

print("Número de pixels com probabilidade de ocorrência menor que", freq_limite, "na imagem não equalizada: ", len(regiao_baixa_freq_n_eq[0]))
print("Número de pixels com probabilidade de ocorrência menor que", freq_limite, "na imagem equalizada: ", len(regiao_baixa_freq_eq[0]))

print("Valores médios dos pixels de baixa probabilidade de ocorrência na imagem não equalizada: ", media_pixel_baixa_freq_n_eq)
print("Valores médios dos pixels de baixa probabilidade de ocorrência na imagem equalizada: ", media_pixel_baixa_freq_eq)
