import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
# Importando e aplicando a função de preprocessamento
from questao2 import preprocess_data


def generate_numerical_features(df, text_column):
    # Vetorizador para DF
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(df[text_column])

    # Vetorizador para TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df[text_column])

    return X_count, X_tfidf, count_vectorizer, tfidf_vectorizer


def plot_top_terms(count_vectorizer, tfidf_vectorizer, X_count, X_tfidf, top_n=10):
    # Obter termos e somatórios
    term_counts = np.array(X_count.sum(axis=0)).flatten()
    term_tfidf = np.array(X_tfidf.sum(axis=0)).flatten()

    # Termos com maior DF
    sorted_indices_df = term_counts.argsort()[::-1][:top_n]
    top_terms_df = np.array(count_vectorizer.get_feature_names_out())[sorted_indices_df]
    top_counts_df = term_counts[sorted_indices_df]

    # Termos com maior TF-IDF
    sorted_indices_tfidf = term_tfidf.argsort()[::-1][:top_n]
    top_terms_tfidf = np.array(tfidf_vectorizer.get_feature_names_out())[sorted_indices_tfidf]
    top_counts_tfidf = term_tfidf[sorted_indices_tfidf]

    # Plotagem
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(top_terms_df, top_counts_df, color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Terms by Document Frequency (DF)')

    plt.subplot(1, 2, 2)
    plt.barh(top_terms_tfidf, top_counts_tfidf, color='lightgreen')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Terms by TF-IDF')

    plt.tight_layout()
    plt.show()


df_CB = pd.read_csv('archive (1)/hateXplain.csv')

df_processed = preprocess_data(df_CB, 'post_tokens')

# Gerando atributos numéricos
X_count, X_tfidf, count_vectorizer, tfidf_vectorizer = generate_numerical_features(df_processed, 'post_tokens')

# Plotando os 10 termos principais
plot_top_terms(count_vectorizer, tfidf_vectorizer, X_count, X_tfidf)
