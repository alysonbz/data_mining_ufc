import re

# 1. **Contagem de Correspondências:**
#    Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.

def cont_word(text):
    regex = r'Python'
    resp = re.findall(regex, text)
    return len(resp)

text = "A linguagem de programação em Python é a mais utilizada do mundo, " \
       "fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print("Resultado primeira questão (Contagem de 'Python'): ", cont_word(text))

# 2. **Validação de E-mail**
def validate_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

# 3. **Extração de Números de Telefone**
def extract_phone_numbers(text):
    pattern = r'\+?\d{1,3}?\s?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}'
    return re.findall(pattern, text)

# 4. **Substituição de Palavras**
def replace_cat_with_dog(text):
    return re.sub(r'gato', 'cachorro', text)

# 5. **Extração de URLs**
def extract_urls(text):
    pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(pattern, text)

# 6. **Verificação de Senha Segura**
def is_secure_password(password):
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(pattern, password) is not None

# 7. **Extração de Palavras**
def extract_words(text):
    return re.findall(r'\w+', text)

# 8. **Validação de Data**
def validate_date(date):
    pattern = r'^\d{2}/\d{2}/\d{4}$'
    return re.match(pattern, date) is not None

# 9. **Extração de Nomes Próprios**
def extract_proper_names(text):
    return re.findall(r'\b[A-Z][a-z]*\b', text)

# 10. **Contagem de Vogais**
def count_vowels(text):
    return len(re.findall(r'[aeiouAEIOU]', text))


# Exemplos de uso:

# 1. Contagem de "Python"
print("Resultado primeira questão (Contagem de 'Python'): ", cont_word(text))

# 2. Validação de E-mail
email = "email.exemplo@dominio.com"
print("Email válido?", validate_email(email))

# 3. Extração de Números de Telefone
text_with_phones = "Me ligue nos números +55 21 98765-4321 ou (11) 1234-5678."
print("Números de telefone extraídos:", extract_phone_numbers(text_with_phones))

# 4. Substituição de Palavras
text_with_cat = "Eu tenho um gato que gosta de brincar com outros gatos."
print("Texto após substituição:", replace_cat_with_dog(text_with_cat))

# 5. Extração de URLs
text_with_urls = "Acesse https://www.google.com e http://www.example.com para mais informações."
print("URLs extraídas:", extract_urls(text_with_urls))

# 6. Verificação de Senha Segura
password = "SenhaForte123!"
print("Senha segura?", is_secure_password(password))

# 7. Extração de Palavras
text_with_words = "Extraia todas as palavras desta frase!"
print("Palavras extraídas:", extract_words(text_with_words))

# 8. Validação de Data
date = "22/09/2024"
print("Data válida?", validate_date(date))

# 9. Extração de Nomes Próprios
text_with_names = "Maria e João foram para São Paulo visitar Carlos."
print("Nomes próprios extraídos:", extract_proper_names(text_with_names))

# 10. Contagem de Vogais
text_with_vowels = "Quantas vogais existem nesta frase?"
print("Contagem de vogais:", count_vowels(text_with_vowels))
