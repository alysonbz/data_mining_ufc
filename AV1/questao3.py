# BIBLIOTECAS E FUNÇÕES --------------------------------
import numpy as np
import matplotlib.pyplot as plt
from questao2 import preprocess_text, corpus
import collections


print("*****************************************")
corpus_prep = corpus.apply(preprocess_text)

print(corpus_prep)

print("*****************************************")


def calculate_df(documentos):
    df = collections.defaultdict(int)
    for doc in documentos:
        for term in set(doc):
            df[term] += 1
    return df

def calculate_tfidf(documentos, df):
    tfidf = {}
    num_documentos = len(documentos)
    for i, doc in enumerate(documentos):
        tf = collections.Counter(doc)
        for term in tf:
            tfidf_score = (tf[term] / len(doc)) * np.log(num_documentos / df[term])
            tfidf[(i, term)] = tfidf_score
    return tfidf

df = calculate_df(corpus_prep)

# Calcula o Term Frequency-Inverse Document Frequency (TF-IDF).
tfidf = calculate_tfidf(corpus_prep, df)

# Plota os 10 termos com o maior TF-IDF.
sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
top_10_tfidf = sorted_tfidf[:10]
top_10_terms_tfidf = [term for (_, term), score in top_10_tfidf]

# Plota os 10 termos com o maior DF.
sorted_df = sorted(df.items(), key=lambda x: x[1], reverse=True)
top_10_df = sorted_df[:10]
top_10_terms_df = [term for term, freq in top_10_df]

# Gráfico dos 10 termos com o maior TF-IDF.
plt.figure(figsize=(12, 6))
plt.barh(top_10_terms_tfidf, [score for (_, term), score in top_10_tfidf])
plt.title("Top 10 Termos com Maior TF-IDF")
plt.xlabel("TF-IDF Score")
plt.gca().invert_yaxis()
plt.show()

# Gráfico dos 10 termos com o maior DF.
plt.figure(figsize=(12, 6))
plt.barh(top_10_terms_df, [freq for term, freq in top_10_df])
plt.title("Top 10 Termos com Maior Document Frequency (DF)")
plt.xlabel("Frequência no Documento")
plt.gca().invert_yaxis()
plt.show()

