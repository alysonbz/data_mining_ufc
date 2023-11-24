import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from questao3 import generate_feature_sets
from questao2 import preprocess_data


df = pd.read_csv('reviews_data.csv')
preprocessed_data = preprocess_data(df)

# Removendo linhas com valores "N/A"
preprocessed_data = preprocessed_data.dropna()

# Dividindo o conjunto de dados em treinamento e teste
X = preprocessed_data['Review']
y = preprocessed_data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando vetores de recursos usando Count Vectorization e TF-IDF
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Calculando o Document Frequency (DF) usando a função da Questão 3
top_10_df, _ = generate_feature_sets(preprocessed_data)
selected_words = top_10_df['Term'].tolist()

# Criando vetores de recursos usando Count Vectorization com palavras selecionadas por DF
count_vectorizer_filtered = CountVectorizer(vocabulary=selected_words)
X_train_count_filtered = count_vectorizer_filtered.transform(X_train)
X_test_count_filtered = count_vectorizer_filtered.transform(X_test)

# Treine um modelo usando Count Vectorization filtrado por DF e Naive Bayes
nb_count_filtered = MultinomialNB()
nb_count_filtered.fit(X_train_count_filtered, y_train)
y_pred_count_filtered = nb_count_filtered.predict(X_test_count_filtered)

# Avalie o desempenho com Count Vectorization filtrado por DF
accuracy_count_filtered = accuracy_score(y_test, y_pred_count_filtered)
print(f'Acurácia com Count Vectorization filtrado por DF e Naive Bayes: {accuracy_count_filtered:.2f}')

# Treine um modelo usando TF-IDF e Naive Bayes
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

# Avalie o desempenho com TF-IDF
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f'Acurácia com TF-IDF e Naive Bayes: {accuracy_tfidf:.2f}')

