
# BIBLIOTECAS E FUNÇÕES --------------------------------

## SKLEARN - MODELOS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
## NUMPY
import numpy as np

## FUNÇÕES DESENVOLVIDAS
from questao2 import preprocess_text, df_emotion
from questao3 import calculate_df, calculate_tfidf
from questao2 import corpus_prep


print("---------------------------------------------------")
X = df_emotion["Comment"].astype(str)
y = df_emotion["Emotion"].astype(str)

print(f'Dataset X: {X =}, \n Y:{y = }')

preprocess_df = preprocess_text(df_emotion).astype(str)
print("----------------------------------------------------")


# Dados de treinamento e teste.
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
print("**********************************************")
print(X_treinamento, X_teste, y_treinamento, y_teste)

print("**********************************************")

bow = CountVectorizer()
X_treinamento_bow = bow.fit_transform(X_treinamento)
X_teste_bow = bow.transform(X_teste)

# Calcula o TF-IDF para os dados de treinamento e teste com a função terceira questão.
df_treinamento = calculate_df(X_treinamento)
tfidf_treinamento = calculate_tfidf(X_treinamento, df_treinamento)

df_teste = calculate_df(X_teste)
tfidf_teste = calculate_tfidf(X_teste, df_teste)

print(df_treinamento,tfidf_treinamento,df_teste,tfidf_teste)

# Valores TF-IDF em matrizes 2D.
X_treinamento_tfidf = np.array(list(tfidf_treinamento.values())).reshape(-1, 1)
X_teste_tfidf = np.array(list(tfidf_teste.values())).reshape(-1, 1)

# Modelo com Naive Bayes.
modelo_treinamento = MultinomialNB()
modelo_treinamento.fit(X_treinamento_tfidf, y_treinamento)

# Previsões nos dados de teste.
previsoes = modelo_treinamento.predict(X_teste_tfidf)

# Desempenho do modelo.
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do modelo TF-IDF: {acuracia:.2f}")
