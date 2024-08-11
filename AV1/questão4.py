import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Carregar o dataset
df = pd.read_csv('sentimentdataset.csv')

# Selecionar as colunas relevantes
df = df[['Text', 'Sentiment']]

# Remover linhas com valores nulos
df.dropna(inplace=True)

# Remover classes com menos de 2 amostras
class_counts = df['Sentiment'].value_counts()
df = df[df['Sentiment'].isin(class_counts[class_counts > 1].index)]

# Verificar a distribuição das classes
print(f"Distribuição ajustada das classes:\n{df['Sentiment'].value_counts()}")

# Aumentar o tamanho do conjunto de teste para evitar o erro de classes > test_size
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.3, random_state=42, stratify=df['Sentiment'])

# Função para vetorização, treinamento e avaliação
def train_and_evaluate(vectorizer, X_train, y_train, X_test, y_test):
    # Vetorizar os textos
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Rebalancear as classes usando SMOTE com k_neighbors=1
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train_vect, y_train)

    # Treinar o modelo Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_res, y_train_res)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test_vect)

    # Avaliar o modelo
    print(f"\nVectorizer: {vectorizer.__class__.__name__}")
    print(classification_report(y_test, y_pred))

# Vetorização usando Bag of Words (CountVectorizer)
count_vectorizer = CountVectorizer()
train_and_evaluate(count_vectorizer, X_train, y_train, X_test, y_test)

# Vetorização usando TF-IDF
tfidf_vectorizer = TfidfVectorizer()
train_and_evaluate(tfidf_vectorizer, X_train, y_train, X_test, y_test)
