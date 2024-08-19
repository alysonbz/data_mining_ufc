'''Instruções

Faça o download do dataset e realize os pré-processamentos adequados. Selecione as colunas que você acredita ser
adequdada de analisar, remova caracteres desnecessários, ajuste o conjunto e tokenize o conjunto,
criando uma função para, inclusive poder ser importada em outras questões.
Plote nessa questão, as cinco primeiras listas de tokens geradas. '''

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Guilherme\Downloads\Nova Pasta\Emotion_classify_Data.csv")

# Função para limpar o texto
def limpar_texto(texto):
    texto = re.sub(r'[^a-zA-Z\s]', '', texto).lower()
    return texto

# Função para tokenizar o texto
def tokenizar_texto(texto):
    tokens = word_tokenize(texto)
    return tokens

# Aplicar as funções limpeza e tokenização
df['Comment'] = df['Comment'].apply(limpar_texto)
df['Tokens'] = df['Comment'].apply(tokenizar_texto)

# Plotar as cinco primeiras listas de tokens
cinco_primeiros_tokens = df['Tokens'].head(5)

fig, eixos = plt.subplots(nrows=5, ncols=1, figsize=(12, 15))

for i, tokens in enumerate(cinco_primeiros_tokens):
    contagem_tokens = Counter(tokens)
    sns.barplot(x=list(contagem_tokens.keys()), y=list(contagem_tokens.values()), ax=eixos[i], palette='rocket')

    eixos[i].set_title(f'Comentário {i + 1}')
    eixos[i].set_xlabel('Tokens')
    eixos[i].set_ylabel('Frequência')
    eixos[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


