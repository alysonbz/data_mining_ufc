'''Instruções

importe a função que você implementou na questão 2 e gere dois conjuntos de atributos numéricos.
O primeiro, baseado no DF e outro baseado no TFIDF.
Plote os 10 termos com maior TF-IDF e os 10 termos com maior DF.
Lembre de implementar em forma de função para que possa ser importada em outra questão.'''

import pandas as pd
*rom sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from questao2 import limpar_texto

df = pd.read_csv(r"C:\Users\Guilherme\Downloads\Nova Pasta\Emotion_classify_Data.csv")

# Função para plotar os 10 Top Termos
def plotar_top_termos(vectorizer, data, top_n=10, titulo=''):
    soma_termos = data.sum(axis=0)
    termos_ordenados = soma_termos.A1.argsort()[::-1][:top_n]
    termos = [vectorizer.get_feature_names_out()[i] for i in termos_ordenados]
    valores = soma_termos.A1[termos_ordenados]

    plt.figure(figsize=(12, 8))
    plt.barh(termos[::-1], valores[::-1], color='pink')
    plt.title(titulo, color='black')
    plt.xlabel('Frequência', color='black')
    plt.ylabel('Termos', color='black')
    plt.show()

# Aplicar a função de limpeza
df['Comment'] = df['Comment'].apply(limpar_texto)

# Importar os vetorizadores
vectorizer_df = CountVectorizer(stop_words='english')
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

# Aplicar vetorizadores
df_train = vectorizer_df.fit_transform(df['Comment'])
tfidf_train = vectorizer_tfidf.fit_transform(df['Comment'])

# Plotar os 10 termos mais frequentes
plotar_top_termos(vectorizer_df, df_train, top_n=10, titulo='10 termos com maior DF')
plotar_top_termos(vectorizer_tfidf, tfidf_train, top_n=10, titulo='10 termos com maior TF-IDF')
