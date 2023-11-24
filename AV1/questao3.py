import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from questao2 import preprocess_data

def generate_feature_sets(data):
    count_vectorizer = CountVectorizer()
    df_matrix = count_vectorizer.fit_transform(data['Review'])
    df_features = count_vectorizer.get_feature_names_out()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Review'])
    tfidf_features = tfidf_vectorizer.get_feature_names_out()

    df_sum = df_matrix.sum(axis=0).A1
    tfidf_sum = tfidf_matrix.sum(axis=0).A1

    df_terms_df = pd.DataFrame({'Term': df_features, 'DF': df_sum})
    tfidf_terms_df = pd.DataFrame({'Term': tfidf_features, 'TF-IDF': tfidf_sum})

    return df_terms_df, tfidf_terms_df

# Carregue o conjunto de dados e faça o preprocessamento
df = pd.read_csv('reviews_data.csv')
preprocessed_data = preprocess_data(df)

# Gere os DataFrames com os termos e valores de DF e TF-IDF
df_terms_df, tfidf_terms_df = generate_feature_sets(preprocessed_data)

# Imprima os 10 principais termos de DF e TF-IDF
print("Top 10 Termos com Maior DF:")
print(df_terms_df.nlargest(10, 'DF'))

print("\nTop 10 Termos com Maior TF-IDF:")
print(tfidf_terms_df.nlargest(10, 'TF-IDF'))
