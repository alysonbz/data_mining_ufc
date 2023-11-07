import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata

df = pd.read_csv(r"C:\Users\joaod\OneDrive\Documentos\Semestre_2023.2\data_minning_ufc\AV1\Data\Consumer Review of Clothing Product\data_amazon.xlsx - Sheet1.csv")
df.dropna(inplace=True)

# Função para pré-processamento e tokenização
def preprocess_text(text):
    # Remover acentuações
    text = ''.join(char for char in unicodedata.normalize('NFKD', text) if unicodedata.category(char) != 'Mn')

    # Tokenização em sentenças
    sentences = sent_tokenize(text.lower())

    # Remover pontuações, números, e lematização
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = []
    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)
        sentence_tokens = [lemmatizer.lemmatize(token) for token in sentence_tokens if token.isalpha()]
        sentence_tokens = [token for token in sentence_tokens if token not in stop_words]
        tokens.extend(sentence_tokens)

    return tokens

# Aplicar a função à coluna 'Review'
df['Review_tokens'] = df['Review'].apply(preprocess_text)

# Aplicar a função à coluna 'Title'
df['Title_tokens'] = df['Title'].apply(preprocess_text)

# Exibir as cinco primeiras listas de tokens de cada coluna
print("Tokens da coluna 'Review':")
print(df['Review_tokens'].head())

print("\nTokens da coluna 'Title':")
print(df['Title_tokens'].head())

# Verificar a contagem de elementos únicos em cada variável após a aplicação da função
print("Contagem de elementos únicos em 'Review_tokens':")
print(df['Review_tokens'].value_counts())

print("\nContagem de elementos únicos em 'Title_tokens':")
print(df['Title_tokens'].value_counts())