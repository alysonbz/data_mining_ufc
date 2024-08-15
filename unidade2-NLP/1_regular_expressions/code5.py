import matplotlib.pyplot as plt
import re
from nltk.tokenize import regexp_tokenize
from src.utils import get_sample_Santo_Graal

# Dividir o script em linhas: lines
holy_grail = get_sample_Santo_Graal()
lines = holy_grail.split('\n')

# Substituir todas as linhas de script para o locutor
pattern = r"[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenizar cada linha: tokenized_lines
tokenized_lines = [regexp_tokenize(s, r'\w+') for s in lines]

# Fazer uma lista de frequência de comprimentos: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plotar um histograma dos comprimentos das linhas
plt.hist(line_num_words, bins=20, color='blue', edgecolor='black')
plt.title('Distribuição do número de palavras por linha')
plt.xlabel('Número de palavras por linha')
plt.ylabel('Frequência')

# Mostrar o gráfico
plt.show()
