import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import get_sample_article

texto = get_sample_article()
# ----------------------- PRÉ-PROCESSAMENTO ---------------------------------
def preprocess_text(text):
    # Tokenização em sentenças
    sentences = sent_tokenize(text.lower())

    # Remover pontuações e números, e lematização
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    preprocessed_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_sentences.append(" ".join(tokens))

    return preprocessed_sentences

# ---------------- FUNÇÕES -----------------------------------
def term_frequency(doc):
    # Calcula a frequência do termo para um documento
    term_freq = Counter(doc)
    total_terms = len(doc)

    # Normaliza pela quantidade total de termos no documento
    tf = {term: freq / total_terms for term, freq in term_freq.items()}

    return tf


def document_frequency(corpus):
    # Calcula a frequência do documento para cada termo no corpus
    df = Counter()

    for doc in corpus:
        # Converte o conjunto de termos em um conjunto para evitar duplicatas
        unique_terms = set(doc)
        df.update(unique_terms)

    total_docs = len(corpus)

    # Normaliza pela quantidade total de documentos no corpus
    df = {term: freq / total_docs for term, freq in df.items()}

    return df


def inverse_document_frequency(corpus, df):
    # Calcula a Frequência Inversa do Documento (IDF) para cada termo no corpus
    idf = {term: math.log(len(corpus) / (1 + freq)) for term, freq in df.items()}

    return idf


def calculate_tfidf(tf, idf):
    # Calcula o TF-IDF multiplicando a Frequência do Termo (TF) pela Frequência Inversa do Documento (IDF)
    tfidf = {term: tf_value * idf[term] for term, tf_value in tf.items()}

    return tfidf

# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------


# Aplicação do pré-processamento
preprocessed_sentences = preprocess_text(texto)

# Cálculos
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sentences)

# Dataframe
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)

# Salvar em CSV
df_tfidf.to_csv('tfidf_matrix.csv', index=False)
