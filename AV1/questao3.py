import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from questao2 import preprocess_text


def generete_numerics_atributes(df, column):
    df['Processed_Text'] = df[column].apply(preprocess_text).apply(lambda x: ' '.join(x))

    # Gerar o conjunto de atributos baseado em DF (Document Frequency)
    count_vectorizer = CountVectorizer()
    df_matrix = count_vectorizer.fit_transform(df['Processed_Text'])
    df_terms = count_vectorizer.get_feature_names_out()
    df_sum = df_matrix.sum(axis=0).A1
    df_term_freq = dict(zip(df_terms, df_sum))

    # Gerar o conjunto de atributos baseado em TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Text'])
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    tfidf_sum = tfidf_matrix.sum(axis=0).A1
    tfidf_term_freq = dict(zip(tfidf_terms, tfidf_sum))

    return df_term_freq, tfidf_term_freq, df_matrix, tfidf_matrix


data = "data_amazon_update.csv"
df = pd.read_csv(data)

df_term_freq, tfidf_term_freq, df_matrix, tfidf_matrix = generete_numerics_atributes(df, column='Review')

# Ordenar e selecionar os 10 termos mais frequentes
top_10_df = sorted(df_term_freq.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_tfidf = sorted(tfidf_term_freq.items(), key=lambda x: x[1], reverse=True)[:10]


def plot_graph_3():
    # Plotar os 10 termos com maior DF
    terms_df, freqs_df = zip(*top_10_df)
    plt.figure(figsize=(10, 6))
    plt.barh(terms_df, freqs_df, color='blue')
    plt.xlabel('Frequency')
    plt.title('Top 10 Terms by Document Frequency (DF)')
    plt.gca().invert_yaxis()
    plt.show()

    # Plotar os 10 termos com maior TF-IDF
    terms_tfidf, freqs_tfidf = zip(*top_10_tfidf)
    plt.figure(figsize=(10, 6))
    plt.barh(terms_tfidf, freqs_tfidf, color='green')
    plt.xlabel('TF-IDF Score')
    plt.title('Top 10 Terms by TF-IDF')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    plot_graph_3()
