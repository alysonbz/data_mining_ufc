# Import necessary modules
from nltk.tokenize import sent_tokenize,word_tokenize
from src.utils import get_sample_Santo_Graal

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)
print(sentences)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
sentence = sentences[3]
tokenized_sent = word_tokenize(sentence)

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)