import pandas as pd
from nltk.tokenize import word_tokenize
import re
import nltk
from collections import Counter
from src.utils import get_english_stop_words
import matplotlib.pyplot as plt

nltk.download('punkt')
data = "data_amazon.xlsx - Sheet1.csv"
df = pd.read_csv(data)

df = df.dropna(subset=['Cons_rating', 'Review'])
df['Cons_rating'] = df['Cons_rating'].apply(lambda x: 0 if x in [1, 2, 3] else 1)
df.to_csv("data_amazon_update.csv")

Reviews = df['Review']


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


df['tokens'] = Reviews.apply(preprocess_text)


def plot_graph(show_plot=True):
    for i, tokens in enumerate(df['tokens'].head(5), 1):
        plt.figure(figsize=(10, 10))
        plt.title(f'Tokens da {i}ª Review')
        plt.bar(range(len(tokens)), [len(token) for token in tokens], tick_label=tokens)
        plt.xticks(rotation=90)

        if show_plot:
            plt.show()


def frequent_tokens(df, top_n=15, show_plot=True):
    # Concatenar todos os tokens em uma lista
    all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
    #  frequência
    token_counts = Counter(all_tokens)
    # tokens mais comuns
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

    if show_plot:
        plt.show()


if __name__ == "__main__":
    plot_graph()
    frequent_tokens(df)
