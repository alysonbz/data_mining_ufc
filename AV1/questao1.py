import pandas as pd
def load_laptop_price():
    return pd.read_csv("AV1/Dataset/laptopPrice.csv")

df = load_laptop_price()

print(df.head())