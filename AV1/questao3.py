import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt


# Baixar as stopwords do NLTK, caso ainda não tenha baixado
nltk.download('stopwords')

def get_english_stop_words():
    return stopwords.words('english')

# Baixar as stopwords do NLTK, caso ainda não tenha baixado
nltk.download('stopwords')

# Função para obter as stopwords em inglês
def get_english_stop_words():
    return stopwords.words('english')

# Função de tokenização previamente implementada
def simple_preprocess_and_tokenize(text):
    # Remover caracteres especiais e números
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Converter o texto para minúsculas
    text = text.lower()

    # Tokenizar o texto (split básico em palavras)
    tokens = text.split()

    # Obter stopwords
    stop_words = set(get_english_stop_words())

    # Remover stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Função para gerar atributos numéricos com DF e TF-IDF e plotar os termos mais relevantes
def generate_and_plot_tfidf_and_df(df, text_column):
    # Inicializando o vetorizador TF-IDF e DF (usando CountVectorizer para DF)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_preprocess_and_tokenize)
    df_vectorizer = CountVectorizer(tokenizer=simple_preprocess_and_tokenize)

    # Gerando as matrizes de TF-IDF e DF
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    df_matrix = df_vectorizer.fit_transform(df[text_column])

    # Extraindo os termos e somando as frequências
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    df_scores = df_matrix.sum(axis=0).A1

    # Associando os termos com suas respectivas pontuações
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    df_terms = df_vectorizer.get_feature_names_out()

    tfidf_ranking = sorted(zip(tfidf_scores, tfidf_terms), reverse=True)[:10]
    df_ranking = sorted(zip(df_scores, df_terms), reverse=True)[:10]

    # Separando os termos e suas pontuações para os gráficos
    tfidf_values, tfidf_labels = zip(*tfidf_ranking)
    df_values, df_labels = zip(*df_ranking)

    # Plotando os termos com maior TF-IDF
    plt.figure(figsize=(10, 5))
    plt.barh(tfidf_labels, tfidf_values, color='blue')
    plt.xlabel('TF-IDF')
    plt.title('Top 10 Terms by TF-IDF')
    plt.gca().invert_yaxis()  # Invertendo o eixo Y para termos mais relevantes no topo
    plt.show()

    # Plotando os termos com maior DF
    plt.figure(figsize=(10, 5))
    plt.barh(df_labels, df_values, color='green')
    plt.xlabel('Document Frequency (DF)')
    plt.title('Top 10 Terms by Document Frequency (DF)')
    plt.gca().invert_yaxis()  # Invertendo o eixo Y para termos mais relevantes no topo
    plt.show()

# Carregar o dataset
file_path = 'C:\\Users\\Amor\\PycharmProjects\\Mine\\data_mining_ufc\\AV1\\reviews_data.csv'
df = pd.read_csv(file_path)

# Gerar e plotar os atributos TF-IDF e DF
generate_and_plot_tfidf_and_df(df, 'Review')
