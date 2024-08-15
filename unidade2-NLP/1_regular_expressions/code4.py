# Import the necessary modules
import re
from nltk.tokenize import TweetTokenizer
from src.utils import get_tweets_sample

# Obter amostra de tweets
tweets = get_tweets_sample()

# Definir um padrão regex para encontrar hashtags: pattern1
pattern1 = r"#\w+"
# Usar o padrão no primeiro tweet da lista de tweets
hashtags = re.findall(pattern1, tweets[0])
print(hashtags)

# Escrever um padrão que corresponda tanto a menções (@) quanto a hashtags
pattern2 = r"(@\w+|#\w+)"
# Usar o padrão no último tweet da lista de tweets
mentions_hashtags = re.findall(pattern2, tweets[-1])
print(mentions_hashtags)

# Usar o TweetTokenizer para tokenizar todos os tweets em uma lista
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)
