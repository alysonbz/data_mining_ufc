import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from questao2 import preprocess_and_tokenize

def generate_features(df):
    # Gerar atributos baseados no DF (Document Frequency)
    vectorizer_df = CountVectorizer(tokenizer=preprocess_and_tokenize, token_pattern=None)
    X_df = vectorizer_df.fit_transform(df['Text'])

    # Gerar atributos baseados no TF-IDF
    vectorizer_tfidf = TfidfVectorizer(tokenizer=preprocess_and_tokenize, token_pattern=None)
    X_tfidf = vectorizer_tfidf.fit_transform(df['Text'])

    return X_df, X_tfidf, vectorizer_df, vectorizer_tfidf


def plot_top_terms(vectorizer, matrix, top_n=10, title="Top Terms"):
    # Obter os termos e suas frequências
    sum_words = matrix.sum(axis=0)
    terms_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    terms_freq = sorted(terms_freq, key=lambda x: x[1], reverse=True)[:top_n]

    # Desempacotar para plotagem
    terms, freqs = zip(*terms_freq)
    plt.figure(figsize=(10, 6))
    plt.barh(terms, freqs, color='blue')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Frequência")
    plt.show()

def main():
    # Carregar o dataset
    df = pd.read_csv("sentimentdataset.csv")

    # Gerar os conjuntos de atributos numéricos
    X_df, X_tfidf, vectorizer_df, vectorizer_tfidf = generate_features(df)

    # Plotar os 10 termos com maior DF e maior TF-IDF
    plot_top_terms(vectorizer_df, X_df, top_n=10, title="Top 10 Terms by DF")
    plot_top_terms(vectorizer_tfidf, X_tfidf, top_n=10, title="Top 10 Terms by TF-IDF")

if __name__ == "__main__":
    main()
