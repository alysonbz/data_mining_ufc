import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import nltk

# Baixar pacotes necessários do NLTK
nltk.download('punkt')

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remover caracteres especiais e pontuação
    text = text.lower()  # Transformar para minúsculas
    tokens = word_tokenize(text)  # Tokenizar
    return tokens

# Função para calcular DF e TF-IDF
def calculate_df_tfidf(df_reviews, vectorizer):
    matrix = vectorizer.fit_transform(df_reviews['review_text'])
    return matrix

# Carregar o dataset
df = pd.read_csv('steam_reviews.csv')

# Verificar as colunas disponíveis no DataFrame
print("Colunas disponíveis:", df.columns)

# Definir a coluna de rótulo
label_column = 'voted_up'

# Remover valores nulos
df = df.dropna(subset=['review_text', label_column])

# Selecionar colunas relevantes
df_reviews = df[['review_text', label_column]]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    df_reviews['review_text'],
    df_reviews[label_column],
    test_size=0.3,
    random_state=42
)

# Inicializar vetorizadores
count_vectorizer = CountVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Calcular DF e TF-IDF
df_matrix_train = calculate_df_tfidf(pd.DataFrame({'review_text': X_train}), count_vectorizer)
df_matrix_test = count_vectorizer.transform(X_test)

tfidf_matrix_train = calculate_df_tfidf(pd.DataFrame({'review_text': X_train}), tfidf_vectorizer)
tfidf_matrix_test = tfidf_vectorizer.transform(X_test)

# Aplicar undersampling para balancear os dados
undersampler = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_resampled_df, y_resampled_df = undersampler.fit_resample(df_matrix_train, y_train)
X_resampled_tfidf, y_resampled_tfidf = undersampler.fit_resample(tfidf_matrix_train, y_train)

from imblearn.combine import SMOTEENN

# Combinação de SMOTE e undersampling (ENN)
smote_enn = SMOTEENN(random_state=42)
X_resampled_df, y_resampled_df = smote_enn.fit_resample(df_matrix_train, y_train)
X_resampled_tfidf, y_resampled_tfidf = smote_enn.fit_resample(tfidf_matrix_train, y_train)

# Treinar e avaliar o classificador com DF balanceado
nb_classifier_df = MultinomialNB()
nb_classifier_df.fit(X_resampled_df, y_resampled_df)
y_pred_df = nb_classifier_df.predict(df_matrix_test)
print("Desempenho com Document Frequency")
print(f"Precisão: {accuracy_score(y_test, y_pred_df):.4f}")
print(classification_report(y_test, y_pred_df))

# Treinar e avaliar o classificador com TF-IDF balanceado
nb_classifier_tfidf = MultinomialNB()
nb_classifier_tfidf.fit(X_resampled_tfidf, y_resampled_tfidf)
y_pred_tfidf = nb_classifier_tfidf.predict(tfidf_matrix_test)
print("Desempenho com TF-IDF")
print(f"Precisão: {accuracy_score(y_test, y_pred_tfidf):.4f}")
print(classification_report(y_test, y_pred_tfidf))
