# Import the necessary modules
import re
from src.utils import get_tweets_sample
from nltk.tokenize import TweetTokenizer

tweets = get_tweets_sample()

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
hashtags = re.findall(pattern1, tweets[0])
print("Hashtags:", hashtags)

# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"(@\w+|#\w+)"
# Use the pattern on the last tweet in the tweets list
mentions_hashtags = re.findall(pattern2, tweets[-1])
print("Mentions and Hashtags:", mentions_hashtags)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(tweet) for tweet in tweets]
print("All Tokens:", all_tokens)
