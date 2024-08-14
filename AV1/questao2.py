import pandas as pd
import re
import matplotlib.pyplot as plt
from src.utils import get_english_stop_words

# Função ajustada de pré-processamento e tokenização sem uso de NLTK
def simple_preprocess_and_tokenize(text):
    # Remover caracteres especiais e números
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Converter o texto para minúsculas
    text = text.lower()

    # Tokenizar o texto (split básico em palavras)
    tokens = text.split()

    # Stopwords básicas
    stop_words = set(get_english_stop_words())
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Carregar o dataset
lv = pd.read_csv('C:\\Users\\Amor\\PycharmProjects\\Mine\\data_mining_ufc\\AV1\\reviews_data.csv')

# Seleção da coluna 'Review' como a mais adequada para a análise de texto
# Esta coluna contém as avaliações que serão processadas e tokenizadas.
# Aplicar a função de pré-processamento e tokenização à coluna 'Review'
lv['tokens'] = lv['Review'].apply(simple_preprocess_and_tokenize)

# Exibir as cinco primeiras listas de tokens geradas
print(lv['tokens'].head())

# Plotar as cinco primeiras listas de tokens
for i, tokens in enumerate(lv['tokens'].head(), start=1):
    plt.figure(figsize=(10, 2))
    plt.title(f'Tokens da Lista {i}')
    plt.barh(range(len(tokens)), [1] * len(tokens), tick_label=tokens)
    plt.show()

from collections import Counter

# Função para plotar a frequência dos tokens
def plot_token_frequencies(tokens, list_index):
    token_freq = Counter(tokens)
    most_common_tokens = token_freq.most_common()

    plt.figure(figsize=(10, 5))
    plt.barh([token for token, _ in most_common_tokens], [freq for _, freq in most_common_tokens])
    plt.xlabel('Frequência')
    plt.ylabel('Tokens')
    plt.title(f'Frequência dos Tokens da Lista {list_index}')
    plt.show()

# Plotar as frequências dos tokens nas cinco primeiras listas
for i, tokens in enumerate(lv['tokens'].head(), start=1):
    plot_token_frequencies(tokens, i)
