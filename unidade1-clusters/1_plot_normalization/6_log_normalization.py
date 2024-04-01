import numpy as np
from src.utils import load_wine_dataset
import pandas as pd
from statistics import variance

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
#print(wine.describe())

## Aplique a função de nomarlização logarítmica na coluna Proline
wine['Proline_log'] = np.log(wine['Proline'])
#
# Print a variância da coluna proline
print(variance(wine['Proline']))

# print a variância da coluna proline normalizada
print(variance(wine['Proline_log']))