import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import get_sample_article

# Carregando as stopwords da língua portuguesa
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Função para calcular TF (Term Frequency) para um documento
def calculate_tf(document):
    word_count = len(document)
    term_freq = {}
    for word in document:
        if word not in term_freq:
            term_freq[word] = document.count(word) / word_count
    return term_freq

# Função para calcular DF (Document Frequency) para um conjunto de documentos
def calculate_df(documents):
    df = {}
    for document in documents:
        unique_words = set(document)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return df

# Função para calcular IDF (Inverse Document Frequency)
def calculate_idf(documents, df):
    idf = {}
    num_documents = len(documents)
    for word, freq in df.items():
        idf[word] = 1 + (num_documents / (1 + freq))
    return idf

# Função para calcular TF-IDF para um conjunto de documentos
def calculate_tfidf(documents, idf):
    tfidf_matrix = []
    for document in documents:
        tf = calculate_tf(document)
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_matrix.append(tfidf)
    return tfidf_matrix

# Seu texto de entrada
texto = get_sample_article()

# Tokenização por sentença
sentences = sent_tokenize(texto)

# Tokenização por palavra e pré-processamento
documents = []
for sentence in sentences:
    tokens = word_tokenize(sentence.lower())  # Converte para minúsculas
    tokens = [token for token in tokens if token.isalpha()]  # Remove caracteres não alfabéticos
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    documents.append(tokens)

# Cálculo de DF e IDF
df = calculate_df(documents)
idf = calculate_idf(documents, df)

# Cálculo de TF-IDF
tfidf_matrix = calculate_tfidf(documents, idf)

# Criando um DataFrame final
df_final = pd.DataFrame(tfidf_matrix)

# Salvando o DataFrame em um arquivo CSV
df_final.to_csv('tfidf_matrix.csv', index=False)

print("DataFrame com matriz TF-IDF salvo como 'tfidf_matrix.csv'")



