import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from src.utils import get_english_stop_words

df = pd.read_csv('/home/luissavio/PycharmProjects/data_mining_ufc/AV1/European Restaurant Reviews.csv')

# Função para limpeza e tokenização do texto
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenizar o texto
    tokens = word_tokenize(text)
    # Remoção de stop words
    stop_words = set(get_english_stop_words())
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Exemplo de aplicação no DataFrame
def process_reviews(df):
    df['Tokens'] = df['Review'].apply(clean_and_tokenize)
    return df

# Processar as reviews
df = process_reviews(df)

# Exibir as cinco primeiras listas de tokens
for i, tokens in enumerate(df['Tokens'].head(5)):
    print(f'Review {i+1}: {tokens}')

# Plotando as cinco primeiras listas de tokens
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(5), [len(tokens) for tokens in df['Tokens'].head(5)], color='blue')
ax.set_yticks(range(5))
ax.set_yticklabels([f'Review {i+1}' for i in range(5)])
ax.set_xlabel('Número de Tokens')
ax.set_title('Número de Tokens nas Cinco Primeiras Reviews')
plt.show()