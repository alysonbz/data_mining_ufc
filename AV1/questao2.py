'''Faça o download do dataset e realize os pré-processamentos adequados. Selecione as colunas que você acredita ser
adequdada de analisar, remova caracteres desnecessários, ajuste o conjunto e tokenize o conjunto, criando uma função para, inclusive
poder ser importada em outras questões. Plote nessa questão, as cinco primeiras listas de tokens geradas. '''

import kaggle
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset="datatattle/covid-19-nlp-text-classification", unzip=True)

'''importando os datasets treino e teste'''
data = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')

'''função para limpeza do testo'''
def texto_processado(data, coluna_texto):
    def limpar_texto(texto):
        texto = re.sub(r"http\S+|www\S+|https\S+", '', texto, flags=re.MULTILINE)
        texto = re.sub(r'\@\w+|\#', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = re.sub(r'\d+', '', texto)
        return texto.lower()
    
    stopword_set = set(stopwords.words('english'))
    
    for coluna in coluna_texto:
        # Aplicar a limpeza de texto
        data[coluna] = data[coluna].apply(limpar_texto)
        
        # Tokenizar o texto limpo
        data[coluna + '_tokens'] = data[coluna].apply(word_tokenize)
        
        # Remover stopwords dos tokens
        data[coluna + '_tokens'] = data[coluna + '_tokens'].apply(
            lambda x: [word for word in x if word not in stopword_set]
        )
    return data


train = data.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])


'''definindo colunas que seram trabalhadas'''
train = texto_processado(train, ['OriginalTweet', 'Sentiment'])


'''visualizar as 5 primeiras linhas do dataset'''

print("Test Dataset Tokens:")
print(train[['OriginalTweet_tokens', 'Sentiment_tokens']].head(5))

train.to_csv("train.csv")

