import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords

# Baixar as stopwords do NLTK, caso ainda não tenha baixado
nltk.download('stopwords')

# Função para obter as stopwords em inglês
def get_english_stop_words():
    return stopwords.words('english')

# Função de tokenização previamente implementada
def simple_preprocess_and_tokenize(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = text.split()

    # Remover stopwords
    stop_words = set(get_english_stop_words())
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Função para gerar atributos numéricos com DF e TF-IDF e aplicar classificação
def classify_and_compare(df, text_column, label_column):
    # Extração de atributos usando DF (Document Frequency)
    df_vectorizer = CountVectorizer(tokenizer=simple_preprocess_and_tokenize)
    df_matrix = df_vectorizer.fit_transform(df[text_column])

    # Extração de atributos usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_preprocess_and_tokenize)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])

    # Divisão dos dados em treinamento e teste (usado para ambas as vetorizações)
    X_train_df, X_test_df, y_train, y_test = train_test_split(df_matrix, df[label_column], test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf = train_test_split(tfidf_matrix, test_size=0.2, random_state=42)[0:2]

    # Aplicando SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    X_train_df, y_train_df = smote.fit_resample(X_train_df, y_train)
    X_train_tfidf, y_train_tfidf = smote.fit_resample(X_train_tfidf, y_train)

    # Classificador Naive Bayes
    clf = MultinomialNB()

    # Treinamento e avaliação usando DF
    clf.fit(X_train_df, y_train_df)
    y_pred_df = clf.predict(X_test_df)
    df_accuracy = accuracy_score(y_test, y_pred_df)
    df_report = classification_report(y_test, y_pred_df)

    # Treinamento e avaliação usando TF-IDF
    clf.fit(X_train_tfidf, y_train_tfidf)
    y_pred_tfidf = clf.predict(X_test_tfidf)
    tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
    tfidf_report = classification_report(y_test, y_pred_tfidf)

    # Resultados
    results = {
        "DF Accuracy": df_accuracy,
        "TF-IDF Accuracy": tfidf_accuracy,
        "DF Classification Report": df_report,
        "TF-IDF Classification Report": tfidf_report,
    }

    return results

# Carregar o dataset
file_path = 'reviews_data.csv'
df = pd.read_csv(file_path)

# Adicionar uma coluna de rótulo (usando 4 como ponto de corte para reviews positivos)
df['label'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Executar a classificação e comparação
results = classify_and_compare(df, 'Review', 'label')

# Exibir os resultados de forma organizada
print("Document Frequency (DF) Classification:")
print("Accuracy:", results["DF Accuracy"])
print("Classification Report:\n", results["DF Classification Report"])
print("\nTF-IDF Classification:")
print("Accuracy:", results["TF-IDF Accuracy"])
print("Classification Report:\n", results["TF-IDF Classification Report"])
