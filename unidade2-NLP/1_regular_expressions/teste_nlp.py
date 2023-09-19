import re

resp = re.match('mineracao', 'mineracao de dados')
print(resp)

resp2 = re.match('dona graca', 'dona graca voce esta bem')
print(resp2)

# resgata a palavra
word_regex = '\w+'
resp3 = re.match(word_regex, 'semana de aula')
print(resp3)

# resgata bla bla bla
word_regex = '\w+'
resp4 = re.match(word_regex, 's emana de aula')
print(resp4)

# ---------- SPLIT -----------
# https://www.pythontutorial.net/python-regex/python-regex-split/

word_regex = '\s+'
resp = re.split(word_regex, 'semana de aula')
print(resp)

word_regex = r'\!'
resp = re.split(word_regex, 'semana quente! de aula')
print(resp)

word_regex = r'[a-z]'
resp = re.split(word_regex, 'Semana Quente! de aula')
print(resp)

word_regex = r'[a-z]\w+'
resp = re.split(word_regex, '4 Semanas Quente! De aula')
print(resp)

# ---------- FINDALL -----------
# https://www.pythontutorial.net/python-regex/python-regex-findall/

word_regex = r'[a-z]\w+'
resp = re.findall(word_regex, '4 Semanas Quente! De aula')
print(resp)

# ---------- SEARCH -----------
#