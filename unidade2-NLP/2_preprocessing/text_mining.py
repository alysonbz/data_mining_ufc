import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math
from src.utils import get_sample_article
import numpy as np

def cal_tf(t, d):
    palavras = d.split()
    frequencia_t = palavras.count(t)
    total_palavras = len(palavras)
    tf = frequencia_t / total_palavras
    return tf

def cal_df(t, d):
    palavras = d.split()
    frequencia_t = palavras.count(t)
    return frequencia_t

def cal_idf(t, d):
    total_d = len(d)
    d_t = sum(1 for documento in d if t in documento)
    idf = math.log10(total_d / (1 + d_t))
    return idf

def tf_idf(tf, idf):
    tfIdf = tf * idf
    return tfIdf

texto = get_sample_article()
sentences = sent_tokenize(texto)
all_tokens = []
stop_words = set(stopwords.words("english"))

for sentence in sentences:
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    all_tokens.extend(tokens)

palavras = list(set(all_tokens))
tf_idf_matrix = np.zeros((len(palavras), len(sentences)))  # Inicializa uma matriz de zeros

for i, palavra in enumerate(palavras):
    for j, sentence in enumerate(sentences):
        tf = cal_tf(palavra, texto)
        df = cal_df(palavra, texto)
        idf = cal_idf(palavra, sentences)
        tfidf = tf_idf(tf, idf)
        tf_idf_matrix[i][j] = tfidf
tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=sentences, index=palavras)
tf_idf_df.to_csv('tfidf_matrix.csv')

