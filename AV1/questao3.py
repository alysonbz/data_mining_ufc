import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# Processa as reviews aplicando a função de limpeza e tokenização
def process_reviews(df):
    df['Tokens'] = df['Review'].apply(clean_and_tokenize)
    return df

# Função para gerar atributos numéricos baseados em DF e TF-IDF
def generate_numerical_features(df):
    # Usando CountVectorizer para DF
    count_vectorizer = CountVectorizer(tokenizer=clean_and_tokenize)
    X_df = count_vectorizer.fit_transform(df['Review'])
    df_terms = count_vectorizer.get_feature_names_out()
    df_freq = X_df.sum(axis=0).A1

    # Usando TfidfVectorizer para TF-IDF
    tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize)
    X_tfidf = tfidf_vectorizer.fit_transform(df['Review'])
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = X_tfidf.sum(axis=0).A1

    return df_terms, df_freq, tfidf_terms, tfidf_scores

# Função para plotar os 10 termos com maior TF-IDF e DF
def plot_top_terms(df_terms, df_freq, tfidf_terms, tfidf_scores):
    # Top 10 termos por DF
    top_df_idx = df_freq.argsort()[-10:][::-1]
    top_df_terms = df_terms[top_df_idx]
    top_df_freq = df_freq[top_df_idx]

    # Top 10 termos por TF-IDF
    top_tfidf_idx = tfidf_scores.argsort()[-10:][::-1]
    top_tfidf_terms = tfidf_terms[top_tfidf_idx]
    top_tfidf_scores = tfidf_scores[top_tfidf_idx]

    # Plotando DF
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(top_df_terms, top_df_freq, color='green')
    plt.title('Top 10 Termos por DF')
    plt.xlabel('Frequência')

    # Plotando TF-IDF
    plt.subplot(1, 2, 2)
    plt.barh(top_tfidf_terms, top_tfidf_scores, color='orange')
    plt.title('Top 10 Termos por TF-IDF')
    plt.xlabel('Pontuação TF-IDF')

    plt.tight_layout()
    plt.show()

df = process_reviews(df)
df_terms, df_freq, tfidf_terms, tfidf_scores = generate_numerical_features(df)
plot_top_terms(df_terms, df_freq, tfidf_terms, tfidf_scores)
