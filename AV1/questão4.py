import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from questao2 import preprocess_data
from questao3 import generate_numerical_features

# Supondo que as funções preprocess_data e generate_numerical_features já foram implementadas e importadas



df_CB = pd.read_csv('archive (1)/hateXplain.csv')
# Preprocessamento dos dados
df_processed = preprocess_data(df_CB, 'post_tokens')

# Codificando as labels
label_encoder = LabelEncoder()
df_processed['label_encoded'] = label_encoder.fit_transform(df_processed['label'])

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_processed['post_tokens'], df_processed['label_encoded'],
                                                    test_size=0.2, random_state=42)


# Função para treinar e avaliar o modelo
def train_and_evaluate_model(X_train, X_test, y_train, y_test, vectorizer):
    # Pipeline para vetorização e classificação
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])

    # Treinando o modelo
    model.fit(X_train, y_train)

    # Avaliando o modelo
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    return report


# Avaliação utilizando CountVectorizer (DF)
df_vectorizer = CountVectorizer()
df_report = train_and_evaluate_model(X_train, X_test, y_train, y_test, df_vectorizer)

# Avaliação utilizando TfidfVectorizer (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_report = train_and_evaluate_model(X_train, X_test, y_train, y_test, tfidf_vectorizer)

# Mostrando os resultados
print("Desempenho usando DF (CountVectorizer):")
print(df_report)
print("\nDesempenho usando TF-IDF (TfidfVectorizer):")
print(tfidf_report)
