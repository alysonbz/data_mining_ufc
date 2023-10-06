from src.utils import get_sample_article
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter

text = """ This quote is on the same wavelength as George Bernard Shaw’s famous “Youth is wasted on the young” sentiment. 
However, Dumbledore, known for having died at the age of 115, takes his thoughts a step further in Harry Potter and the Order of the Phoenix. 
Harry’s 15 years old in the book and concerned about a number of things, including where he stands with his friends, being a fifth-year student at Hogwarts, and He-Who-Must-Not-Be-Named. 
With his young mind occupied with so much, Dumbledore’s statement about youth makes all the more sense."""

sentences = sent_tokenize(text)
tokens = [word_tokenize(s) for s in sentences]

c = [Counter(token) for token in tokens]

print(f"{sentences = }\n{tokens =}\n{c = }")