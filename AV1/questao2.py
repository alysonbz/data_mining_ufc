# PACKAGES -------------------------------------------------------------------------------------------------------------

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# FUNÇÃO ---------------------------------------------------------------------------------------------------------------

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(text):
    if isinstance(text, str):
        # Remove caracteres não alfanuméricos e transforma em minúsculas
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Tokenização
        tokens = word_tokenize(text)

        # Redução das palavras à raiz e remoção de palavras com menos de 2 caracteres
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens if len(word) > 2]

        # Remoção de stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    else:
        tokens = []
    return tokens


# TESTE DA FUNÇÃO ------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/steam_reviews.csv')

reviews = df['review_text']

tokenized_reviews = reviews.apply(preprocess_and_tokenize)

print('Primeiras 5 observações:\n', reviews.head(5))

print('Primeiras 5 listas de tokens:')
for i in range(5):
    print(f'Review {i + 1} Tokens:')
    print(tokenized_reviews.iloc[i])
