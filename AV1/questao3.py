# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from AV1.questao2 import preprocess_and_tokenize


# FUNÇÃO ---------------------------------------------------------------------------------------------------------------

def generate_text_attributes(data, method): # a função recebe o df e o método desejado
    if method == "DF":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data) # conta a frequência de palavras
        df = pd.DataFrame(X.toarray(),  # converte as contagens em um dataframe
                          columns=vectorizer.get_feature_names_out())
        return df
    elif method == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(data) # calcula os valores TF-IDF pra cada termo
        tfidf = pd.DataFrame(tfidf_matrix.toarray(), # converte as contagens em um dataframe
                             columns=tfidf_vectorizer.get_feature_names_out())
        return tfidf
    else:
        return None


# TESTE DA FUNÇÃO ------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/steam_reviews.csv')

df = df.head(7000)

reviews = df['review_text']

tokenized_reviews = [preprocess_and_tokenize(text) for text in reviews]

tokenized_reviews = [' '.join(tokens) for tokens in tokenized_reviews]

# Obter os 10 termos com maior DF
df_terms = generate_text_attributes(tokenized_reviews, method="DF")
top_df_terms = df_terms.sum().sort_values(ascending=False).head(10) # soma as contagens de palavras

# Obter os 10 termos com maior TF-IDF
tfidf_terms = generate_text_attributes(tokenized_reviews, method="TF-IDF")
top_tfidf_terms = tfidf_terms.mean().sort_values(ascending=False).head(10) # média dos valores TF-IDF

top_df_terms.plot.bar()
plt.title('DF: top 10 termos')
plt.show()

top_tfidf_terms.plot.bar()
plt.title('TF-IDF: top 10 termos')
plt.show()
