# text_mining.py

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter
from src.utils import get_sample_article


# Função para pré-processamento
def preprocess_text(text):
    # Tokenização por sentença
    sentences = sent_tokenize(text)
    # Tokenização por palavra
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    return tokenized_sentences, sentences


# Função para calcular TF (Term Frequency)
def calculate_tf(tokenized_sentences):
    tf = []
    for sentence in tokenized_sentences:
        count = Counter(sentence)
        tf_sentence = {word: count[word] / len(sentence) for word in count}
        tf.append(tf_sentence)
    return tf


# Função para calcular DF (Document Frequency)
def calculate_df(tokenized_sentences):
    df = defaultdict(int)
    num_docs = len(tokenized_sentences)
    for sentence in tokenized_sentences:
        unique_words = set(sentence)
        for word in unique_words:
            df[word] += 1
    return df


# Função para calcular IDF (Inverse Document Frequency)
def calculate_idf(df, num_docs):
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log(num_docs / (freq + 1))  # +1 para evitar divisão por zero
    return idf


# Função para calcular TF-IDF
def calculate_tfidf(tf, idf):
    tfidf = []
    for sentence_tf in tf:
        tfidf_sentence = {word: sentence_tf.get(word, 0) * idf.get(word, 0) for word in sentence_tf}
        tfidf.append(tfidf_sentence)
    return tfidf


# Função principal para executar a mineração de texto
def main():
    # Obter o texto do artigo de amostra
    text = get_sample_article()

    # Pré-processar o texto
    tokenized_sentences, original_sentences = preprocess_text(text)

    # Calcular TF
    tf = calculate_tf(tokenized_sentences)

    # Calcular DF
    df = calculate_df(tokenized_sentences)

    # Calcular IDF
    num_docs = len(tokenized_sentences)
    idf = calculate_idf(df, num_docs)

    # Calcular TF-IDF
    tfidf = calculate_tfidf(tf, idf)

    # Criar DataFrame
    df_tfidf = pd.DataFrame(tfidf).fillna(0)
    df_tfidf['Sentence'] = original_sentences

    # Salvar o DataFrame em um arquivo CSV
    df_tfidf.to_csv('tfidf_matrix.csv', index=False)
    print("DataFrame salvo em 'tfidf_matrix.csv'.")


if __name__ == "__main__":
    main()
