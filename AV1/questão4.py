import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from questao2 import texto_processado
from questao3 import gerar_features
from sklearn.preprocessing import LabelEncoder

# Importar os datasets
train = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')

# Aplicar a função de pré-processamento
train = texto_processado(train, ['OriginalTweet'])

le = LabelEncoder()
y = le.fit_transform(train['Sentiment'])

# Remover colunas desnecessárias
train = train.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])

# Gerar atributos numéricos usando DF e TF-IDF
X_df, X_tfidf, _, _ = gerar_features(train, 'OriginalTweet_tokens')

# Confirmar correspondência entre amostras
assert X_tfidf.shape[0] == train['Sentiment'].shape[0], "O número de amostras em X_tfidf não corresponde ao número de amostras em train['Sentiment']"

# Separando os dados de treino e teste
X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Definir os modelos
model_df = LogisticRegression()
model_tfidf = LogisticRegression()

# Treinando com os dados de DF
model_df.fit(X_train_df, y_train)

# Treinando com os dados de TF-IDF
model_tfidf.fit(X_train_tfidf, y_train)

# Prevendo os resultados
y_pred_df = model_df.predict(X_test_df)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Avaliando o desempenho
print("Desempenho usando DF:")
print(classification_report(y_test, y_pred_df))

print("\nDesempenho usando TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))
