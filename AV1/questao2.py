import pandas as pd
from nltk.tokenize import word_tokenize
import re
from src.utils import get_english_stop_words
import matplotlib.pyplot as plt
from collections import Counter

# Carregando o dataset
df = pd.read_csv("Tweets.csv")

#Removendo valores nulos
df = df.dropna(subset=["sentiment", "text"])

# Salvando o datatset
df.to_csv("Tweets_atualizado.csv")
print(df.head())

#Selecionando Reviews
Textos = df["text"]

# Função para limpar texto e tokenizar
def preprocess_text(text):
    # Remover todos os caracteres que não são letras ou espaços
    text = re.sub(r'[^A-Za-z\s]', '', str(text).lower())
    # Tokenizar o texto
    tokens = word_tokenize(text)
    # Remover stop words
    stop_words = set(get_english_stop_words())
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Função para plotar as palavras mais frequentes
def plot_most_frequent_tokens(df, top_n=20):
    # Concatenar todos os tokens em uma única lista
    all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]

    # Contar a frequência de cada token
    token_counts = Counter(all_tokens)

    # Obter os tokens mais comuns
    most_common_tokens = token_counts.most_common(top_n)

    # Separar tokens e suas contagens
    tokens, counts = zip(*most_common_tokens)

    # Plotar o gráfico de barras
    plt.figure(figsize=(12, 8))
    plt.bar(tokens, counts, color='orange')
    plt.xlabel('Tokens')
    plt.ylabel('Frequência')
    plt.title(f'Top {top_n} Tokens Mais Frequentes')
    plt.xticks(rotation=90)
    plt.show()

#Função para plotar as palavras mais frequentes
def plot_graph():
    for i, tokens in enumerate(df['tokens'].head(5), 1):
        plt.figure(figsize=(10, 8))
        plt.title(f'Tokens da {i}ª Textos')
        plt.bar(range(len(tokens)), [len(token) for token in tokens], tick_label=tokens)
        plt.xticks(rotation=90)
        plt.show()


def display_first_five_token_lists(df):
    # Mostrar as cinco primeiras listas de tokens
    for i, tokens in enumerate(df['tokens'].head(5), 1):
        print(f"\nTokens da {i}ª Textos:")
        token_df = pd.DataFrame(tokens, columns=['Token'])
        print(token_df)


# Garantindo que o plot seja executado apenas quando a questão 2 for executada
if __name__ == "__main__":
    df['tokens'] = Textos.apply(preprocess_text)
    print(df["tokens"].head())
    display_first_five_token_lists(df)
    plot_most_frequent_tokens(df)