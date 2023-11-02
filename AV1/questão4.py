# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from AV1.questao2 import preprocess_and_tokenize
from AV1.questao3 import generate_text_attributes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# MODELO -------------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/steam_reviews.csv')

df = df.head(7000)

reviews = df['review_text']
labels = df['voted_up']

# Pré-processamento dos dados
preprocessed_data = [preprocess_and_tokenize(text) for text in reviews]

preprocessed_data = [' '.join(tokens) for tokens in preprocessed_data]

# Extração de atributos no método "DF"
df_attributes = generate_text_attributes(preprocessed_data, method="DF")

# Extração de atributos no método "TF-IDF"
tfidf_attributes = generate_text_attributes(preprocessed_data, method="TF-IDF")

# DF - Divisão dos dados em conjuntos de treinamento e teste
DFX_train, DFX_test, DFy_train, DFy_test = train_test_split(df_attributes, labels,
                                                            test_size=0.2,
                                                            random_state=12)

# TFIDF - Divisão dos dados em conjuntos de treinamento e teste
TFIDFX_train, TFIDFX_test, TFIDFy_train, TFIDFy_test = train_test_split(tfidf_attributes, labels,
                                                                        test_size=0.2,
                                                                        random_state=12)

# Treinamento e avaliação do modelo com "DF"
model_df = RandomForestClassifier(n_estimators=100, random_state=12)
model_df.fit(DFX_train, DFy_train)
predictions_df = model_df.predict(DFX_test)
accuracy_df = accuracy_score(DFy_test, predictions_df)

# Treinamento e avaliação do modelo com "TF-IDF"
model_tfidf = RandomForestClassifier(n_estimators=100, random_state=12)
model_tfidf.fit(TFIDFX_train, TFIDFy_train)
predictions_tfidf = model_tfidf.predict(TFIDFX_test)
accuracy_tfidf = accuracy_score(TFIDFy_test, predictions_tfidf)

# Comparação de desempenho
print("Acurácia com atributos DF:", accuracy_df)
print("Acurácia com atributos TF-IDF:", accuracy_tfidf)
