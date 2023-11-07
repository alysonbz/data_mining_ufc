import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from questao2 import preprocess_tokenize


df = pd.read_csv('restaurant.csv')

# Análise da coluna Reviews que contém avaliações de restaurante
reviews = df['Review']

# Removendo linhas com valores nan
reviews = reviews.dropna()

# Função para extrair atributos
# CountVectorizer considera apenas a frequência das palavras
def extract_df_features(reviews):
    vectorizer = CountVectorizer(tokenizer=preprocess_tokenize) # o objeto CountVectorizer está sendo usado para realizar a contagem de palavras
    X = vectorizer.fit_transform(reviews) # o texto é transf em uma matriz de contagem de palavras
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()) # matriz em Dataframe
    return df

# Função para extrair atributos com base no TF-IDF
# Considera-se a importância relativa das palavras com base no TF-IDF.
def extract_tfidf_features(reviews):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_tokenize) # TfidfVectorizer está sendo usado para calcular os valores TF-IDF das palavras
    X = vectorizer.fit_transform(reviews)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df

# Aplicando as funções
df_df_features = extract_df_features(reviews)
df_tfidf_features = extract_tfidf_features(reviews)

# 10 termos com maior TF-IDF
print("Os 10 termos com maior TF-IDF são:")
print(df_tfidf_features.sum().nlargest(10))

# 10 termos com maior DF
print("Os 10 termos com maior df são:")
print(df_df_features.sum().nlargest(10))

# RESULTADO:  A quantidade de palavras positivas, como "good", "pleasant", "great",
# "amazing", "courteous", "helpful" e "enjoyed", sugerem que muitos aspectos do restaurante
# são avaliados positivamente.