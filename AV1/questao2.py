import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


df = pd.read_csv('reviews_data.csv')  # Substitua 'starbucks_reviews_dataset.csv' pelo caminho real do seu arquivo CSV.


selected_columns = ['Rating', 'Review']  # Escolha as colunas que deseja analisar
df = df[selected_columns]


df['Review'] = df['Review'].str.replace('[^a-zA-Z\s]', '').str.lower()


df['Tokens'] = df['Review'].apply(lambda x: word_tokenize(x))


def preprocess_data(data):
    data = data[selected_columns]
    data['Review'] = data['Review'].str.replace('[^a-zA-Z\s]', '').str.lower()
    data['Tokens'] = data['Review'].apply(lambda x: word_tokenize(x))
    return data


preprocessed_data = preprocess_data(df)

# Exiba as cinco primeiras listas de tokens
for i in range(5):
    print(preprocessed_data['Tokens'][i])
