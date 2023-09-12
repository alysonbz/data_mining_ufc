import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

# Print the statistical characteristics of the dataset wine
print(wine.describe())

## Apply the logarithmic normalization to the 'Proline' column
wine['Proline'] = np.log(wine['Proline'])

# Print the variance of the 'Proline' column
print(wine['Proline'].var())

# Print the variance of the normalized 'Proline' column
print(wine['Proline'].var())
