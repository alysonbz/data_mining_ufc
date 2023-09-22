import re

#1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

# print(cont_word(text))

def valid_email(email):
    email_base = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,6}"
    email_valido = re.fullmatch(email_base, email) is not None
    return f"Endereço de e-mail {'válido' if email_valido else 'inválido'}"

def valid_number(numeros):

    numero_telefone_regex = r"((?:0|\+55)\d{2})?\d{2,3}[ -]?\d{2,5}[ -]?\d{4}"

    match = re.search(numero_telefone_regex, numeros)
    if match is None:
        raise ValueError("Nenhum número de telefone encontrado")

    return match.group(0)

# print(valid_number("Meu némero de telefone é"
#                   " : (85)991931047"))

def substituir_palavras(texto, palavra_original, palavra_substituta):
    palavra_original_regex = r"\b" + palavra_original + r"\b"
    texto_substituido = re.sub(palavra_original_regex, palavra_substituta, texto)
    return texto_substituido

# print(substituir_palavras("O gato é um animal de estimação", "gato", "cachorro"))#

def extrair_urls(texto):
    url_regex = r"(https?://\S+)"
    urls = re.findall(url_regex, texto)

    return urls

# print(extrair_urls("Este texto contém a URL https://www.google.com"))

def validar_senha_segura(senha):
    senha_segura_regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[@#$%^&+=])[a-zA-Z0-9@#$%^&+=]{8,}$"

    match = re.fullmatch(senha_segura_regex, senha)

    if match is not None:
        return "Senha forte"
    else:
        return "Senha fraca"

# print(validar_senha_segura("abc123@776IMNA%"))

def extrair_palavras(texto):
    palavra_regex = r"\w+"
    palavras = re.findall(palavra_regex, texto)

    return palavras


print(extrair_palavras("Este texto contém várias palavras"))
