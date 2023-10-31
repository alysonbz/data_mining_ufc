import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
import matplotlib.pyplot as plt
import re
file_path = r'C:\Users\mateu\Downloads\archive (2)\Corona_NLP_train.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
colunas_principais = ['OriginalTweet']
data = data[colunas_principais]

def stop_words_function(df, column_name, new_column_name):
    stop_words = set(stopwords.words('english'))
    df[new_column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    return df


def clean_function(df, column_text):
    # Remove URLs
    df[column_text + '_cleaned'] = df[column_text].apply(lambda x: re.sub(r'http\S+', '', x))
    # Remove caracteres não alfanuméricos e mantém espaços em branco
    df[column_text + '_cleaned'] = df[column_text + '_cleaned'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    # Converte o texto para minúsculas
    df[column_text + '_cleaned'] = df[column_text + '_cleaned'].str.lower()
    # Tratar valores nulos (NaN) antes da normalização
    df[column_text + '_cleaned'] = df[column_text + '_cleaned'].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', errors='ignore').decode('utf-8'))
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    df[column_text + '_cleaned_no_stopwords'] = df[column_text + '_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    return df

def tokenization (df, column_name, new_column_name):
    df[new_column_name] = df[column_name].map(lambda x: word_tokenize(x))
    return df

def preprocess(df, column_text):
    df = clean_function(df, column_text)
    df = tokenization(df, column_text + '_cleaned_no_stopwords', column_text + '_tokenized')
    return df

df = pd.read_csv(file_path, encoding='ISO-8859-1') # Carregue seu DataFrame aqui
column_text = 'OriginalTweet'  # Substitua pelo nome da coluna de texto

df = preprocess(df, column_text)
def extract_first_five_tokens(df, column_name):
    first_five_tokens = df[column_name + '_tokenized'].head(5)
    return first_five_tokens
first_five_tokens = extract_first_five_tokens(df, column_text)
for i, tokens in enumerate(first_five_tokens):
    print(f"Tokens da linha {i + 1}: {tokens}")
def plot_tokens(tokens, title, ax):
    ax.plot(tokens, marker='o', markersize=5, linestyle='-', linewidth=2)
    ax.set_title(title)
    ax.grid(True)

tokens_list = [
    ['menyrbie', 'philgahan', 'chrisitv'],
    ['advice', 'talk', 'neighbours', 'family', 'exchange', 'phone', 'numbers', 'create', 'contact', 'list', 'phone', 'numbers', 'neighbours', 'schools', 'employer', 'chemist', 'gp', 'set', 'online', 'shopping', 'accounts', 'poss', 'adequate', 'supplies', 'regular', 'meds', 'order'],
    ['coronavirus', 'australia', 'woolworths', 'give', 'elderly', 'disabled', 'dedicated', 'shopping', 'hours', 'amid', 'covid19', 'outbreak'],
    ['food', 'stock', 'one', 'empty', 'please', 'dont', 'panic', 'enough', 'food', 'everyone', 'take', 'need', 'stay', 'calm', 'stay', 'safe', 'covid19france', 'covid19', 'covid19', 'coronavirus', 'confinement', 'confinementotal', 'confinementgeneral'],
    ['ready', 'go', 'supermarket', 'covid19', 'outbreak', 'im', 'paranoid', 'food', 'stock', 'litteraly', 'empty', 'coronavirus', 'serious', 'thing', 'please', 'dont', 'panic', 'causes', 'shortage', 'coronavirusfrance', 'restezchezvous', 'stayathome', 'confinement']
]

fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, (tokens, ax) in enumerate(zip(tokens_list[:5], axs)):
    plot_tokens(tokens, f'List {i + 1}', ax)

plt.tight_layout()


plt.show()
