# PACKAGES -------------------------------------------------------------------------------------------------------------

from src.utils import get_sample_article
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import string


# FUNÇÕES --------------------------------------------------------------------------------------------------------------

# ---> TF (Term Frequency)
def calculate_tf(documents):
    tf_list = []

    for document in documents:
        word_count = Counter(document.split())  # Conta quantas vezes cada palavra aparece no documento

        # Calcula a frequência de cada palavra no documento
        tf = {word: count / len(document.split()) for word, count in word_count.items()}

        tf_list.append(tf)

    return tf_list


# ---> DF (Document Frequency)
def calculate_df(documents):
    df_list = Counter()

    for document in documents:
        unique_words = set(document.split())  # Palavras únicas no documento
        df_list.update(unique_words)  # Atualiza o contador de DF com as palavras únicas do documento

    return df_list


# ---> IDF (Inverse Document Frequency)
def calculate_idf(documents, df):
    idf_dict = {}

    for word, count in df.items():
        idf_dict[word] = np.log(len(documents) / (1 + count))  # Calcula o IDF para cada palavra usando a fórmula

    return idf_dict


# ---> TF-IDF (Term Frequency-Inverse Document Frequency)
def calculate_tfidf(documents, idf):
    tfidf_dict = []

    for document in documents:
        tfidf = {}
        word_count = Counter(document.split())
        total_words = len(document.split())

        for word, count in word_count.items():
            tfidf[word] = (count / total_words) * idf[word]  # Calcula o TF-IDF para cada palavra usando as frequências
                                                             # de palavras e o IDF
        tfidf_dict.append(tfidf)

    return tfidf_dict


# PRÉ-PROCESSAMENTO ----------------------------------------------------------------------------------------------------

article = get_sample_article()

sentences = sent_tokenize(article)  # Tokenização por sentença

stopwords = set(stopwords.words("english"))  # Stopwords em inglês

PS = PorterStemmer()  # Reduz as palavras à sua raiz

pp_sentences = []

for sentence in sentences:
    tokens = word_tokenize(sentence)  # Tokeniza cada sentença em palavras

    # Reduz palavras à raiz e converte para minúsculas, removendo pontuação e caracteres especiais
    tokens = [PS.stem(token.lower()) for token in tokens if token not in string.punctuation and token.isalpha()]

    tokens = [token for token in tokens if token not in stopwords]  # Remove stopwords

    pp_sentences.append(" ".join(tokens))  # Recria a sentença após o pré-processamento


# TESTE ----------------------------------------------------------------------------------------------------------------

TF = calculate_tf(pp_sentences)
#print(pd.DataFrame(TF))

DF = calculate_df(pp_sentences)
#print(DF)

IDF = calculate_idf(pp_sentences, DF)
#print(IDF)

TFIDF = calculate_tfidf(pp_sentences, IDF)
TFIDF_df = pd.DataFrame(TFIDF)
#print(TFIDF_df)

TFIDF_df.to_csv("TFIDF_df.csv", index=False)
