import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from questao2 import preprocess
from sklearn.metrics import  classification_report

train_file_path = "C:/Users/mateu/Downloads/archive (2)/Corona_NLP_train.csv"
test_file_path = "C:/Users/mateu/Downloads/archive (2)/Corona_NLP_test.csv"

train_data = pd.read_csv(train_file_path, encoding='ISO-8859-1')
test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')


text_column = 'OriginalTweet'
label_column = 'Sentiment'


train_data = preprocess(train_data, text_column)
test_data = preprocess(test_data, text_column)

X_train, X_val, y_train, y_val = train_test_split(train_data[text_column], train_data[label_column], test_size=0.2, random_state=42)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_val_count = count_vectorizer.transform(X_val)
X_test_count = count_vectorizer.transform(test_data[text_column])

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(test_data[text_column])

naive_bayes_count = MultinomialNB()
naive_bayes_tfidf = MultinomialNB()

naive_bayes_count.fit(X_train_count, y_train)
naive_bayes_tfidf.fit(X_train_tfidf, y_train)

y_pred_val_count = naive_bayes_count.predict(X_val_count)
y_pred_val_tfidf = naive_bayes_tfidf.predict(X_val_tfidf)

accuracy_count = accuracy_score(y_val, y_pred_val_count)
accuracy_tfidf = accuracy_score(y_val, y_pred_val_tfidf)

print("Acurácia com Count Vectorizer:", accuracy_count)
print("Acurácia com TF-IDF Vectorizer:", accuracy_tfidf)

y_pred_test_count = naive_bayes_count.predict(X_test_count)
y_pred_test_tfidf = naive_bayes_tfidf.predict(X_test_tfidf)

# Avalie o desempenho nos dados de teste
accuracy_test_count = accuracy_score(test_data[label_column], y_pred_test_count)
accuracy_test_tfidf = accuracy_score(test_data[label_column], y_pred_test_tfidf)

print("Acurácia nos dados de teste com Count Vectorizer:", accuracy_test_count)
print("Acurácia nos dados de teste com TF-IDF Vectorizer:", accuracy_test_tfidf)

print("Relatório de Classificação com Count Vectorizer:")
print(classification_report(test_data[label_column], y_pred_test_count))

print("Relatório de Classificação com TF-IDF Vectorizer:")
print(classification_report(test_data[label_column], y_pred_test_tfidf))