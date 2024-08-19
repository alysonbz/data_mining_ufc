'''Instruções

Utilizando as funções immplementadas nas questões anteriores,
aplique uma classificação com o algoritmo apropriado, comparando
o desempenho com as duas formas de extração de atributos implementadas.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from questao2 import limpar_texto

df = pd.read_csv(r"C:\Users\Guilherme\Downloads\Nova Pasta\Emotion_classify_Data.csv")

# Aplicar a função de limpeza
df['Comment'] = df['Comment'].apply(limpar_texto)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['Emotion'], test_size=0.3, random_state=53)

# Inicializar os vetorizadores
vectorizer_df = CountVectorizer(stop_words='english')
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

# Transformar os dados de treino e teste
df_train = vectorizer_df.fit_transform(X_train.values)
df_test = vectorizer_df.transform(X_test.values)

tfidf_train = vectorizer_tfidf.fit_transform(X_train.values)
tfidf_test = vectorizer_tfidf.transform(X_test.values)

# Inicializar o classificador Naive Bayes
naive_bayes = MultinomialNB()

# Treinar e avaliar o modelo usando CountVectorizer (DF)
naive_bayes.fit(df_train, y_train)
y_pred_df = naive_bayes.predict(df_test)
accuracy_df = accuracy_score(y_test, y_pred_df)

# Treinar e avaliar o modelo usando TfidfVectorizer (TF-IDF)
naive_bayes.fit(tfidf_train, y_train)
y_pred_tfidf = naive_bayes.predict(tfidf_test)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

# Imprimir as acurácias
print(f'Acurácia usando CountVectorizer (DF): {accuracy_df:.4f}')
print(f'Acurácia usando TfidfVectorizer (TF-IDF): {accuracy_tfidf:.4f}')

