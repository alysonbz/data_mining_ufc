# Import the necessary modules
from nltk.tokenize import regexp_tokenize, TweetTokenizer
from src.utils import get_tweets_sample

# Obtenha a amostra de tweets
tweets = get_tweets_sample()

# Defina um padrão regex para encontrar hashtags: pattern1
pattern1 = r"#\w+"
# Use o padrão no primeiro tweet da lista de tweets
hashtags = regexp_tokenize(tweets[0], pattern1)
print(hashtags)

# Escreva um padrão que combina tanto menções (@) quanto hashtags
pattern2 = r"[@#]\w+"
# Use o padrão no último tweet da lista de tweets
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print(mentions_hashtags)

# Use o TweetTokenizer para tokenizar todos os tweets em uma lista
tknzr = TweetTokenizer()
all_tokens = [token for t in tweets for token in tknzr.tokenize(t)]
print(all_tokens)
