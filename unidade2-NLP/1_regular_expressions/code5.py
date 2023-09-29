import matplotlib.pyplot as plt
import re
from nltk.tokenize import regexp_tokenize
from src.utils import get_sample_Santo_Graal

# Split the script into lines: lines
holy_grail = get_sample_Santo_Graal()
lines = holy_grail.split('\n')  # Split the script into lines using newline character

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', line) for line in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(line, pattern=r'\s+', gaps=True) for line in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(line) for line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words, bins=20, edgecolor='k')
plt.title('Histograma')
plt.xlabel('Numero de linhas')
plt.ylabel('Frequencia')

# Show the plot
plt.show()
