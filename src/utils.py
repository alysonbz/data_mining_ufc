import pandas as pd
import matplotlib as plt


def load_wine_dataset():
    wine = pd.read_csv('../datasets/wine.csv')
    return wine

def load_pokemon_dataset():
    return [9, 6, 2, 3, 1, 7, 1, 6, 1, 7, 23, 26, 25, 23, 21, 23, 23, 20, 30, 23] ,\
        [8, 4, 10, 6, 0, 4, 10, 10, 6, 1, 29, 25, 30, 29, 29, 30, 25, 27, 26, 30]

def loadpokemon_dataset_df():
    return pd.read_csv('../datasets/pokemon.csv')

def load_fifa_dataset():
    return pd.read_csv('../datasets/fifa_18_dataset.csv')

def load_comic_con_dataset():
    return pd.read_csv('../datasets/comic_con.csv')


