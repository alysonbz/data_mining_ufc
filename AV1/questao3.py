import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from questao2 import preprocess
from questao2 import file_path
def generate_numeric_attributes(df, text_column):
    preprocessed_data = preprocess(df, text_column)

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(preprocessed_data['OriginalTweet_cleaned_no_stopwords'])
    count_features = count_vectorizer.get_feature_names_out()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_data['OriginalTweet_cleaned_no_stopwords'])
    tfidf_features = tfidf_vectorizer.get_feature_names_out()

    count_means = count_matrix.mean(axis=0).A1

    top_10_count_indices = count_means.argsort()[-10:][::-1]
    top_10_count_terms = [count_features[i] for i in top_10_count_indices]

    tfidf_means = tfidf_matrix.mean(axis=0).A1

    top_10_tfidf_indices = tfidf_means.argsort()[-10:][::-1]
    top_10_tfidf_terms = [tfidf_features[i] for i in top_10_tfidf_indices]

    print("Top 10 Termos com Maior Count:")
    print(top_10_count_terms)

    print("Top 10 Termos com Maior TF-IDF:")
    print(top_10_tfidf_terms)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(top_10_count_terms, count_means[top_10_count_indices], color='blue')
    plt.title("Top 10 Termos com Maior Count")
    plt.xlabel("Count")
    plt.ylabel("Termo")

    plt.subplot(1, 2, 2)
    plt.barh(top_10_tfidf_terms, tfidf_means[top_10_tfidf_indices], color='green')
    plt.title("Top 10 Termos com Maior TF-IDF")
    plt.xlabel("TF-IDF")
    plt.ylabel("Termo")

    plt.tight_layout()
    plt.show()

    return count_matrix, tfidf_matrix

df = pd.read_csv(file_path, encoding='ISO-8859-1')
text_column = 'OriginalTweet'
count_matrix, tfidf_matrix = generate_numeric_attributes(df, text_column)
