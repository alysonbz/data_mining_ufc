import pandas as pd
import re
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


df_CB = pd.read_csv('archive (1)/hateXplain.csv')

print(df_CB.head())


def preprocess_data(df, text_column):
    df = df[['post_id', 'label', 'target', text_column]]
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', str(x).lower()))
    df['tokens'] = df[text_column].apply(word_tokenize)
    return df


# Aplicando a função de preprocessamento
df_processed = preprocess_data(df_CB, 'post_tokens')

print(df_processed[['post_id', 'tokens']].head())

# Plotando as cinco primeiras listas de tokens em subplots
fig, axes = plt.subplots(5, 1, figsize=(12, 10))

for i, (tokens, ax) in enumerate(zip(df_processed['tokens'].head(), axes)):
    ax.bar(range(len(tokens)), [1] * len(tokens))
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation='vertical')
    ax.set_title(f'Tokens for post_id {df_processed["post_id"].iloc[i]}')

plt.tight_layout()
plt.show()
