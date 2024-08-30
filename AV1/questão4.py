import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords


# Carregar o dataset
df = pd.read_csv('sentimentdataset.csv')

# Selecionar as colunas relevantes
df = df[['Text', 'Sentiment']]

# Remover linhas com valores nulos
df.dropna(inplace=True)

# Remover classes com menos de 4 amostras
class_counts = df['Sentiment'].value_counts()
df = df[df['Sentiment'].isin(class_counts[class_counts >= 4].index)]

# Verificar a distribuição das classes
print(f"Distribuição ajustada das classes (5 primeiras):\n{df['Sentiment'].value_counts().head()}")

# Dividir o dataset em treino e teste
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

    # Avaliar o modelo (somente nas 5 primeiras classes)
    print(f"\nVectorizer: {vectorizer.__class__.__name__}")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Exibir apenas as 5 primeiras classes
    report_df = pd.DataFrame(report).transpose().head(5)
    print(report_df)

# Função para comparar diferentes formas de vetorização
def compare_vectorizers(X_train, y_train, X_test, y_test):
    # Vetorização usando Bag of Words (CountVectorizer)
    count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    train_and_evaluate(count_vectorizer, X_train, y_train, X_test, y_test)

    # Vetorização usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    train_and_evaluate(tfidf_vectorizer, X_train, y_train, X_test, y_test)

# Aplicar o algoritmo protetor e comparar as formas de vetorização
compare_vectorizers(X_train, y_train, X_test, y_test)
