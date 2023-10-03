import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Fazer o download de recursos necessários do NLTK
nltk.download("punkt")
nltk.download("stopwords")


# Função para calcular o TF (Term Frequency)
def calculate_tf(tokens):
    tf = {}
    total_tokens = len(tokens)

    for token in tokens:
        if token in tf:
            tf[token] += 1
        else:
            tf[token] = 1

    for token, freq in tf.items():
        tf[token] = freq / total_tokens

    return tf


# Função para calcular o DF (Document Frequency)
def calculate_df(sentences):
    df = {}

    for sentence in sentences:
        tokens = set(word_tokenize(sentence.lower()))
        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1

    return df


# Função para calcular o IDF (Inverse Document Frequency)
def calculate_idf(df, total_documents):
    idf = {}

    for token, freq in df.items():
        idf[token] = 1 + (total_documents / (1 + freq))

    return idf


# Função para calcular o TF-IDF
def calculate_tf_idf(tf, idf):
    tfidf = {}

    for token, tf_value in tf.items():
        tfidf[token] = tf_value * idf[token]

    return tfidf


# Texto de exemplo
text = "Coloque aqui o seu texto de exemplo. Você pode incluir várias sentenças ou parágrafos. Lembre-se de que este é apenas um exemplo."

# Tokenização por sentença
sentences = sent_tokenize(text)

# Pré-processamento: remoção de stop words e tokenização
stop_words = set(stopwords.words("portuguese"))
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
filtered_sentences = [[word for word in tokens if word.isalpha() and word not in stop_words] for tokens in
                      tokenized_sentences]

# Cálculo do TF para cada sentença
tf_values = [calculate_tf(tokens) for tokens in filtered_sentences]

# Cálculo do DF para todo o corpus
df_values = calculate_df(sentences)

# Total de documentos
total_documents = len(sentences)

# Cálculo do IDF
idf_values = calculate_idf(df_values, total_documents)

# Cálculo do TF-IDF para cada sentença
tfidf_values = [calculate_tf_idf(tf, idf_values) for tf in tf_values]

# Criação do DataFrame
df_tfidf = pd.DataFrame(tfidf_values)

# Salvar o DataFrame em formato CSV
df_tfidf.to_csv("tfidf_matrix.csv", index=False)

# Exibição do DataFrame
print(df_tfidf)
