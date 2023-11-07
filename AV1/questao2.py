import nltk
nltk.download('stopwords') # download da lista de stopwords em inglês
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('restaurant.csv')

# Análise da coluna Reviews que contém avaliações de restaurante
reviews = df['Review']

# pré-processamento e tokenização
def preprocess_tokenize(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z]', ' ', text)   # Removendo caracteres especiais e números
        text = text.lower()  # Convertendo o texto para palavras minúsculas
        tokens = word_tokenize(text)    # Tokenização
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remoção das stopwords
        return tokens
    else:
        return []  # lista vazia se não for uma string

# Aplicando a função a cada elemento da coluna 'Review'
df['Tokens'] = reviews.apply(preprocess_tokenize)

# Cinco primeiras listas de tokens
for i in range(5):
    print(f"Tokens da lista {i + 1}:")
    print(df.loc[i, 'Tokens'])
    print()
