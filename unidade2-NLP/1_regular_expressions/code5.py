import matplotlib.pyplot as plt
import re
from nltk.tokenize import regexp_tokenize
from src.utils import get_sample_Santo_Graal

# Split the script into lines: lines
holy_grail = get_sample_Santo_Graal()
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "([A-Z]{2,}(?:\s#\d+)?:)"
lines = [re.sub(pattern, '', line).strip() for line in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(line, pattern="\w+") for line in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words, bins=20)
plt.xlabel('Quantidade de palavras em uma linha')
plt.ylabel('Frequência')
plt.title('Distribuição dos Comprimentosem "Monty Python e o Santo Graal"')
plt.show()

