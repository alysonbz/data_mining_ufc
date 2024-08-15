import re
from nltk.tokenize import sent_tokenize
from src.utils import get_sample_Santo_Graal

# Carregar o texto de amostra
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Procurar a primeira ocorrência de "coconuts" em scene_one: match
match = re.search(r"coconuts", scene_one)

# Imprimir os índices de início e fim do match
print(match.start(), match.end())
print(match)

# Escrever uma expressão regular para buscar qualquer coisa entre colchetes: pattern1
pattern1 = r"\[.*?\]"

# Usar re.search para encontrar o primeiro texto entre colchetes
brackets_text = re.search(pattern1, scene_one)
print(brackets_text.group())

# Encontrar a notação de script no início da quarta sentença e imprimi-la
pattern2 = r"^\[.*?\]"
script_notation = re.search(pattern2, sentences[3])
print(script_notation.group())
