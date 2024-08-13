# Import Counter and word_tokenize
from nltk.tokenize import word_tokenize

from NLP.src.nlp_utils import get_sample_article

article = get_sample_article()

# Tokenize the article: tokens
tokens = ___(__)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [__ for t in ___]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = __(__)

# Print the 10 most common tokens
print(___)