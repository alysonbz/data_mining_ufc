from src.utils import get_sample_article
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

texto = get_sample_article()
# Tokenização por sentença
sentencas = sent_tokenize(texto)


# Calcular TF-IDF
def calcular_tfidf(sentencas):
    documentos = []  # Lista para armazenar as sentenças

    vectorizer = TfidfVectorizer()

    for sentenca in sentencas:
        documentos.append(sentenca)  # Cada sentença é um documento

    tfidf_matrix = vectorizer.fit_transform(documentos)
    feature_names = vectorizer.get_feature_names_out()

    # Criar um df com a matriz TF-IDF
    df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)

    # Remover todos os valores do dataframe que são 0
    df_tfidf = df_tfidf.loc[:, (df_tfidf != 0).any(axis=0)]

    return df_tfidf


# Calcular TF-IDF
df_tfidf = calcular_tfidf(sentencas)

# Mostrar o df
print(df_tfidf)

# Salvar em .csv
df_tfidf.to_csv('tfidf_matrix.csv', index=False)


