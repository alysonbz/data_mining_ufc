'''import pandas as pd
import re
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(text):
    # Remover caracteres especiais e pontuação
    text = re.sub(r'[^\w\s]', '', text)
    # Transformar para minúsculas
    text = text.lower()
    # Tokenizar
    tokens = word_tokenize(text)
    return tokens

# Carregar o dataset
df = pd.read_csv('steam_reviews.csv')

df = df.dropna()
# Selecionar colunas relevantes
df_reviews = df[['review_text']]

# Aplicar pré-processamento e tokenização
df_reviews['tokens'] = df_reviews['review_text'].apply(preprocess_and_tokenize)

# Exibir as cinco primeiras listas de tokens
print("Cinco primeiras listas de tokens:")
for i, tokens in enumerate(df_reviews['tokens'][:5]):
    print(f"Revisão {i+1}: {tokens}")

# Plotar as cinco primeiras listas de tokens
plt.figure(figsize=(10, 5))
for i, tokens in enumerate(df_reviews['tokens'][:5]):
    plt.plot(tokens, label=f'Revisão {i+1}')
plt.legend()
plt.title('Cinco Primeiras Listas de Tokens')
plt.xlabel('Tokens')
plt.ylabel('Frequência')
plt.show()
'''

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(text):
    # Remover caracteres especiais e pontuação
    text = re.sub(r'[^\w\s]', '', text)
    # Transformar para minúsculas
    text = text.lower()
    # Tokenizar
    tokens = word_tokenize(text)
    return tokens

# Carregar o dataset
df = pd.read_csv('steam_reviews.csv')

# Remover valores nulos
df = df.dropna()

# Selecionar colunas relevantes
df_reviews = df[['review_text']]

# Aplicar pré-processamento e tokenização
df_reviews['tokens'] = df_reviews['review_text'].apply(preprocess_and_tokenize)

# Função para criar gráfico das 10 palavras mais frequentes
def plot_top_tokens(tokens, title):
    token_counts = Counter(tokens)
    most_common_tokens = token_counts.most_common(10)
    tokens, counts = zip(*most_common_tokens)
    plt.bar(tokens, counts)
    plt.title(title)
    plt.xlabel('Tokens')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)

# Criar gráficos das 10 palavras mais frequentes para as 5 primeiras revisões
plt.figure(figsize=(14, 8))
for i, tokens in enumerate(df_reviews['tokens'][:5]):
    plt.subplot(2, 3, i+1)  # Subplot para múltiplos gráficos
    plot_top_tokens(tokens, f'Revisão {i+1}')

plt.tight_layout()
plt.show()

# Análise 1: Número médio de tokens por revisão
average_tokens = df_reviews['tokens'].apply(len).mean()

# Análise 2: Frequência das palavras mais comuns em todas as revisões
all_tokens = [token for tokens in df_reviews['tokens'] for token in tokens]
most_common_overall = Counter(all_tokens).most_common(10)

# Análise 3: Revisões com mais de 50 tokens
reviews_over_50_tokens = df_reviews[df_reviews['tokens'].apply(len) > 50].shape[0]

print(average_tokens, most_common_overall, reviews_over_50_tokens)


# Exibir as cinco primeiras listas de tokens
print("Cinco primeiras listas de tokens:")
for i, tokens in enumerate(df_reviews['tokens'][:5]):
    print(f"Revisão {i+1}: {tokens}\n")

# Análise 1: Número médio de tokens por revisão
average_tokens = df_reviews['tokens'].apply(len).mean()
print(f"Número médio de tokens por revisão: {average_tokens:.2f}\n")

# Análise 2: Frequência das palavras mais comuns em todas as revisões
all_tokens = [token for tokens in df_reviews['tokens'] for token in tokens]
most_common_overall = Counter(all_tokens).most_common(10)
print("Palavras mais comuns em todas as revisões:")
for word, count in most_common_overall:
    print(f"{word}: {count}")

# Análise 3: Revisões com mais de 50 tokens
reviews_over_50_tokens = df_reviews[df_reviews['tokens'].apply(len) > 50].shape[0]
print(f"\nNúmero de revisões com mais de 50 tokens: {reviews_over_50_tokens}")