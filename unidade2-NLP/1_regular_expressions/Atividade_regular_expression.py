import re

#1. **Contagem de Correspondências:**
#Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.
print('Primeira atividade')
def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

text = "A linguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print(cont_word(text))


# Validação de email
print('Segunda atividade')
def valid_email(email):
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(regex, email))

text = "livia@example.com"
text2 = "livia_synth@example"

print(valid_email(text2))

# Extração de números de telefone
print('Terceira atividade')
def extract_phone_numbers(text):
    regex = r'\(?\d{2}\)?\s?\d{4,5}-\d{4}|\d{4}-\d{3}-\d{3}|\d{4}-\d{4}'
    return re.findall(regex, text)

text = "Meu telefone é (11) 98364-4362"
text2 = "Meu telefone é 9866-4523"
print(extract_phone_numbers(text))

# Substituição de palavra
print('Quarta atividade')
def replace_word(text, old, new):
    return re.sub(old, new, text)

text = "olá, meu nome é Livia. E estou passeando com meu gato."
print(replace_word(text, "gato", "cachorro"))

# Extração de URL
print('Quinta atividade')
def extract_urls(text):
    regex = r'(https?://[^\s]+)'
    return re.findall(regex, text)

text = "Acesse o site https://www.A_melhor_de_todas.com e veja as novidades em https://www.Livia10.com"
print(extract_urls(text))

# Verificação de senha segura
print('Sexta atividade')
def is_secure_password(password):
    regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(regex, password))

text = "Livia"
text2 = "Livia@"
print(is_secure_password(text2))

# Extração de palavra
print('Sétima atividade')
def extract_words(text):
    regex = r'\w+'
    return re.findall(regex, text)

text = "Olá, meu nome é Livia. E estou passeando com meu gato."
print(extract_words(text))

# Validação de data
print('Oitava atividade')
def valid_date(date):
    regex = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(regex, date))

text = "2022-12-31"
text2 = "202-122-31"
print(valid_date(text))

# Extração de nomes próprios
print('Nona atividade')
def extract_names(text):
    regex = r'[A-Z][a-z]+(?: [A-Z][a-z]+)*'
    return re.findall(regex, text)

text = "olá, meu nome é Livia. E estou passeando com a Maverick."
print(extract_names(text))

# Contagem de vogais
print('Décima atividade')
def count_vowels(text):
    regex = r'[aeiouAEIOU]'
    return len(re.findall(regex, text))

text = "UFC estava de greve."
print(count_vowels(text))
