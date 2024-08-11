import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Baixar pacotes necessários para tokenização e remoção de stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Função de pré-processamento e tokenização
def preprocess_and_tokenize(text):
    # Convertendo o texto para minúsculas
    text = text.lower()
    # Removendo URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Removendo menções e hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Removendo números e pontuações
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenização
    tokens = word_tokenize(text)
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def main():
    # Carregar o dataset
    df = pd.read_csv("sentimentdataset.csv")

    # Selecionar colunas relevantes para análise
    selected_columns = ['Text']
    df = df[selected_columns]

    # Aplicar a função de pré-processamento e tokenização
    df['tokens'] = df['Text'].apply(preprocess_and_tokenize)

    # Exibir as cinco primeiras listas de tokens gerados
    print(df['tokens'].head())

if __name__ == "__main__":
    main()
