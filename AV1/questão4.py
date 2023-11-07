import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from questao3 import extract_tfidf_features, extract_df_features

df = pd.read_csv('restaurant.csv')

# Removendo linhas nan na coluna Review e Rating
df.dropna(subset=['Review', 'Rating'], inplace=True)

# Selecionando as colunas
reviews = df['Review']
labels = df['Rating']

# Função questão 3
def extract_df_features(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df

# Função questão 3
def extract_tfidf_features(reviews):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df

# Função para realizar a classificação
def classify(reviews, labels, feature_extraction):
    if feature_extraction == "DF":
        features = extract_df_features(reviews) # utiliza a extração de recursos com CountVectorizer
    elif feature_extraction == "TFIDF":
        features = extract_tfidf_features(reviews) # utiliza a extração de recursos com TfidfVectorizer.
    else:
        raise ValueError("Método de extração de características não suportado")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # classificador Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# classificação com base em DF
accuracy_df = classify(reviews, labels, "DF")
print(f"Acurácia com base em DF: {accuracy_df:.2f}")

# classificação com base em TF-IDF
accuracy_tfidf = classify(reviews, labels, "TFIDF")
print(f"Acurácia com base em TF-IDF: {accuracy_tfidf:.2f}")
