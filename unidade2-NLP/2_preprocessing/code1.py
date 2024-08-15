# Importar Counter e word_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize
from src.utils import get_sample_article

# Obter amostra do artigo
article = get_sample_article()

# Tokenizar o artigo: tokens
tokens = word_tokenize(article)

# Converter os tokens para minúsculas: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Criar um Counter com os tokens em minúsculas: bow_simple
bow_simple = Counter(lower_tokens)

# Imprimir os 10 tokens mais comuns
print(bow_simple.most_common(10))
