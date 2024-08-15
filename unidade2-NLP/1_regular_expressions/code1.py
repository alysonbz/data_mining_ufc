import re

# Definir um padrão para corresponder ao final de uma sentença
sentence_endings = r"[.!?]"

# Dividir my_string nos finais de sentença e imprimir o resultado
my_string = "Let's write RegEx!  Won't that be fun?  " \
            "I sure think so.  Can you find 4 sentences? " \
            "Or perhaps, all 19 words?"
print(re.split(sentence_endings, my_string))

# Encontrar todas as palavras capitalizadas em my_string e imprimir o resultado
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Dividir my_string nos espaços e imprimir o resultado
spaces = r"\s+"
print(re.split(spaces, my_string))

# Encontrar todos os dígitos em my_string e imprimir o resultado
digits = r"\d+"
print(re.findall(digits, my_string))
