# Import the necessary modules

# Import the necessary library
from src.utils import get_tweets_sample
from nltk.tokenize import TweetTokenizer  # Make sure you have NLTK installed

# Get a sample of tweets
tweets = get_tweets_sample()

# Define a regex pattern to find hashtags: pattern1
import re
pattern1 = r"#\w+"

# Use the pattern on the first tweet in the tweets list
hashtags = re.findall(pattern1, tweets[0])
print(hashtags)

# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"[@#]\w+"

# Use the pattern on the last tweet in the tweets list
mentions_hashtags = re.findall(pattern2, tweets[-1])
print(mentions_hashtags)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(tweet) for tweet in tweets]
print(all_tokens)
