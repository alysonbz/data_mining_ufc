from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def extract_attributes_met_pixels(images):
    dados = []
    for image in images:
        dados.append({'C1_mean': np.mean(image[:, :, 0]),
                      'C2_mean': np.mean(image[:, :, 1]),
                      'C3_mean': np.mean(image[:, :, 2]),
                      'C1_std': np.std(image[:, :, 0]),
                      'C2_std': np.std(image[:, :, 1]),
                      'C3_std': np.std(image[:, :, 2]),
                      'C1_median': np.median(image[:, :, 0]),
                      'C2_median': np.median(image[:, :, 1]),
                      'C3_median': np.median(image[:, :, 2])})
    return pd.DataFrame(dados)


def extract_attributes_met_hist(images):
    dados = []
    for image in images:
        C1_hist = plt.hist(image[:, :, 0].ravel(), bins=16)
        C2_hist = plt.hist(image[:, :, 1].ravel(), bins=16)
        C3_hist = plt.hist(image[:, :, 2].ravel(), bins=16)
        dados.append({'C1_mean': np.mean(C1_hist[0]),
                      'C2_mean': np.mean(C2_hist[0]),
                      'C3_mean': np.mean(C3_hist[0]),
                      'C1_std': np.std(C1_hist[0]),
                      'C2_std': np.std(C2_hist[0]),
                      'C3_std': np.std(C3_hist[0]),
                      'C1_median': np.median(C1_hist[0]),
                      'C2_median': np.median(C2_hist[0]),
                      'C3_median': np.median(C3_hist[0])})
    return pd.DataFrame(dados)


def extract_attributes_hist(images):
    dados = []
    for image in images:
        C1_hist = plt.hist(image[:, :, 0].ravel(), bins=16)
        C2_hist = plt.hist(image[:, :, 1].ravel(), bins=16)
        C3_hist = plt.hist(image[:, :, 2].ravel(), bins=16)
        c1_c2 = np.concatenate((C1_hist[0], C2_hist[0]))
        c2 = np.concatenate((c1_c2, C3_hist[0]))
        dados.append(c2)
    return np.array(dados)


def extract_attributes_pixels(images):
    dados = []
    for image in images:
        print('i')
        C1_hist = image[:, :, 0].flatten()
        C2_hist = image[:, :, 1].flatten()
        C3_hist = image[:, :, 2].flatten()
        dados.append(list(C1_hist) + list(C2_hist) + list(C3_hist))
    return np.array(dados)
