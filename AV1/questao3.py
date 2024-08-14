import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remover caracteres especiais e pontuação
    text = text.lower()  # Transformar para minúsculas
    tokens = word_tokenize(text)  # Tokenizar
    return tokens

# Função para calcular DF e TF-IDF
def calculate_df_tfidf(df_reviews):
    # Vetorizador de Frequência (Document Frequency - DF)
    count_vectorizer = CountVectorizer()
    df_matrix = count_vectorizer.fit_transform(df_reviews['review_text'])

    # Vetorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_reviews['review_text'])

    # Retornar os vetores e os nomes dos termos
    return count_vectorizer, tfidf_vectorizer, df_matrix, tfidf_matrix


# Função para plotar os 10 termos mais frequentes em DF e TF-IDF
# Função para plotar os 10 termos mais frequentes em DF e TF-IDF
# Função para plotar os 10 termos mais frequentes em DF e TF-IDF
def plot_top_terms(vectorizer, matrix, top_n=10, title=""):
    # Somar as ocorrências de cada termo
    term_frequencies = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    # Criar um DataFrame para facilitar a manipulação dos dados
    term_df = pd.DataFrame({'term': terms, 'frequency': term_frequencies})

    # Selecionar os top N termos
    top_terms = term_df.nlargest(top_n, 'frequency')

    # Plotar
    plt.figure(figsize=(10, 6))
    plt.barh(top_terms['term'], top_terms['frequency'])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('Frequência')
    plt.ylabel('Termos')
    plt.show()

# Função para exibir os top termos
def print_top_terms(vectorizer, matrix, top_n=10, title=""):
    # Somar as ocorrências de cada termo
    term_frequencies = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    # Criar um DataFrame para facilitar a manipulação dos dados
    term_df = pd.DataFrame({'term': terms, 'frequency': term_frequencies})

    # Selecionar os top N termos
    top_terms = term_df.nlargest(top_n, 'frequency')

    # Exibir os resultados
    print(title)
    print(top_terms)
    print()

# Carregar o dataset
df = pd.read_csv('steam_reviews.csv')

# Remover valores nulos
df = df.dropna()

# Selecionar colunas relevantes
df_reviews = df[['review_text']]

# Aplicar pré-processamento e tokenização
df_reviews['tokens'] = df_reviews['review_text'].apply(preprocess_and_tokenize)

# Calcular DF e TF-IDF
count_vectorizer, tfidf_vectorizer, df_matrix, tfidf_matrix = calculate_df_tfidf(df_reviews)

# Exibir os 10 termos mais frequentes em DF
print_top_terms(count_vectorizer, df_matrix, top_n=10, title="Top 10 Termos com Maior DF")

# Exibir os 10 termos mais frequentes em TF-IDF
print_top_terms(tfidf_vectorizer, tfidf_matrix, top_n=10, title="Top 10 Termos com Maior TF-IDF")

# Plotar os 10 termos mais frequentes em DF
plot_top_terms(count_vectorizer, df_matrix, top_n=10, title="Top 10 Termos com Maior DF")

# Plotar os 10 termos mais frequentes em TF-IDF
plot_top_terms(tfidf_vectorizer, tfidf_matrix, top_n=10, title="Top 10 Termos com Maior TF-IDF")