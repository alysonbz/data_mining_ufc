import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

-
#print as caractéristicas estatísticas do dataset wine
print(wine.describe())

## Aplique a função de nomarlização logarítmica na coluna Proline
wine['Proline_norm'] = np.log(wine['Proline'])


print('\nvariância:', wine['Proline'].var())
print('variância normalizada', wine['Proline_norm'].var())