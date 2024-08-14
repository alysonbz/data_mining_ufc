import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_english_stop_words

df = pd.read_csv('/home/luissavio/PycharmProjects/data_mining_ufc/AV1/European Restaurant Reviews.csv')

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenizar o texto
    tokens = word_tokenize(text)
    # Remoção de stop words
    stop_words = set(get_english_stop_words())
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def process_reviews(df):
    df['Tokens'] = df['Review'].apply(clean_and_tokenize)
    return df


def fit_and_transform_vectorizer(vectorizer, X_train, X_test):
    vectorizer.fit(X_train)
    X_train_transformed = vectorizer.transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    return X_train_transformed, X_test_transformed


def train_and_evaluate(X_train, X_test, y_train, y_test, method):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Results for {method.upper()}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 60)


# Preparando o alvo (Sentiment)
y = df['Sentiment']

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['Review'], y, test_size=0.2, random_state=42)

# Criando os vetorizadores
count_vectorizer = CountVectorizer(tokenizer=clean_and_tokenize)
tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize)

# Ajustando e transformando os conjuntos de treino e teste para DF
X_train_df, X_test_df = fit_and_transform_vectorizer(count_vectorizer, X_train, X_test)
train_and_evaluate(X_train_df, X_test_df, y_train, y_test, 'df')

# Ajustando e transformando os conjuntos de treino e teste para TF-IDF
X_train_tfidf, X_test_tfidf = fit_and_transform_vectorizer(tfidf_vectorizer, X_train, X_test)
train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, 'tfidf')