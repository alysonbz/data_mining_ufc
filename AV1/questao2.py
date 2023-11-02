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
        # Removendo caracteres
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # Tokenização
        tokens = word_tokenize(text)
        # Transformando tudo em minúsculo
        tokens = [word.lower() for word in tokens]
        # Retirando possíveis espaços do início e do final da palavra
        tokens = [word.strip() for word in tokens]
        # Reduzindo palavras à raiz
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        # Removendo palavras menores que 2 caracteres
        tokens = [word for word in tokens if len(word) > 2]
        # Removendo stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    else:
        tokens = []
    return tokens


# TESTE DA FUNÇÃO ------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/steam_reviews.csv')

df = df.head(7000)

reviews = df['review_text']

tokenized_reviews = reviews.apply(preprocess_and_tokenize)

print('Primeiras 5 observações:\n', reviews.head(5))

print('Primeiras 5 listas de tokens:')
for i in range(5):
    print(f'Review {i + 1} Tokens:')
    print(tokenized_reviews.iloc[i])
