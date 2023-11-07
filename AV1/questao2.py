
#BIBLIOTECAS E FUNÇÕES -----------------------------------------------------------------------------
import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize,sent_tokenize

def load_emotion_classify():
    return pd.read_csv("/home/bbmq/Documentos/mineracao_dados/data_minning_ufc/AV1/Dataset/Emotion_classify_Data.csv")

df_emotion = load_emotion_classify()
corpus = df_emotion["Comment"].astype(str)
print("----------------------------------")

print("DATAFRAME :\n\n")
print(corpus)

print("----------------------------------")
print("TEXTO PRÉ PROCESSADO")

def preprocess_text(text):
    # Remoção de emojis.
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Remoção de caracteres indesejados (exceto letras e espaços).
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub("\n",'',text)

    # Padronizando o texto para palavras ninúsculas
    text = text.lower()

    # Tokenização o texto em palavras.
    tokens = nltk.word_tokenize(text)

    # Remoção as stopwords (palavras comuns, como "é", "de", "em", etc.).
    stop_words = set(stopwords.words('english'))  # Stopwords do dataset das palavras em inglês.
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Junção.
    cleaned_text = ' '.join(tokens)
    return tokens

corpus_prep = corpus.apply(preprocess_text)
print(corpus_prep)


print("----------------------------------")
print("PLOT 5 PRIMEIRAS LISTAS DE TOKENS:\n")

print(corpus_prep[:5])
def plot_tokens(tokens, title, ax):
    ax.plot(tokens, marker='o', markersize=5, linestyle='-', linewidth=2)
    ax.set_title(title)
    ax.grid(True)

fig, axs = plt.subplots(2, 3, figsize=(15, 6))


for i, (tokens, ax) in enumerate(zip(corpus_prep[:3], axs[0])):
    plot_tokens(tokens, f'List {i + 1}', ax)


for i, (tokens, ax) in enumerate(zip(corpus_prep[3:], axs[1])):
    plot_tokens(tokens, f'List {i + 4}', ax)


plt.tight_layout()

plt.show()