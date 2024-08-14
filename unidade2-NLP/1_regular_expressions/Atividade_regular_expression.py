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

print("resultado primeira questão: ", cont_word(text))

#2. **Validação de E-mail:**
# Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares.

def check(email):
    regex = r'^[\w-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
    if re.fullmatch(regex, email):
        print("Email Válido")
    else:
        print("Email Inválido!")

email = "sheldasouza21@gmail.com"
print(check(email))

#3. **Extração de Números de Telefone:**
# Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares.

def extrair_numeros_telefone(texto):
    regex = r'\(?\b\d{2}\)?\s?-?\d{4,5}-?\d{4}\b'
    numeros_telefone = re.findall(regex, texto)

    return numeros_telefone


texto2 = """
(11) 91234-5678
11-91234-5678
11912345678
(11) 3456-7890
+55 85 91234-5678
123457
32 765
"""
numeros = extrair_numeros_telefone(texto2)
print(numeros)
# 4. **Substituição de Palavras:**
#Crie uma função que substitua todas as ocorrências de "gato" por "cachorro" em um texto usando expressões regulares.
def substituir_gato_por_cachorro(texto):
    regex = r'\bgato\b'
    texto_modificado = re.sub(regex, 'cachorro', texto)

    return texto_modificado

texto3 = "Atirei o pau no gato mas o gato não morreu."
texto_modificado = substituir_gato_por_cachorro(texto3)
print(texto_modificado)
#5. **Extração de URLs:**
# Escreva um programa que extraia todas as URLs de um texto usando expressões regulares.
def extrair_urls(texto):
    regex = r'https?://(?:www\.)?\S+(?:/|(?=\s))'
    urls = re.findall(regex, texto)

    return urls
texto4 = """
https://www.exemplo.com
http://exemplo.org.br
Acesse: https://www.frivjogosonline.com.br/
"""
urls = extrair_urls(texto4)
print(urls)

#6. **Verificação de Senha Segura:**
#Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais.
def validar_senha(senha):
    regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    if re.match(regex, senha):
        print("Senha válida")
    else:
        print("Senha inválida!")

senha1 = "Souza@123"
senha2 = "sonhOa123"

print(validar_senha(senha1))
print(validar_senha(senha2))

# 7. **Extração de Palavras:**
# Escreva uma função que extraia todas as palavras de uma string usando expressões regulares.
def extrair_palavras(texto):
    regex = r'\b\w+\b'
    palavras = re.findall(regex, texto)

    return palavras

texto5 = "um, dois, três, somos trigêmeas sim!"
palavras = extrair_palavras(texto5)
print(palavras)

# 8. **Validação de Data:**
# Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares.
def validar_data(data):
    regex = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/([0-9]{4})$'

    if re.match(regex, data):
        print("data válida.")
    else:
        print("data inválida!")
data1 = "29/02/2024"
data2 = "15/08/2023"

print(validar_data(data1))
print(validar_data(data2))

#9. **Extração de Nomes Próprios:**
# Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando expressões regulares.
def extrair_nomes_proprios(texto):
    regex = r'\b[A-Z][a-z]*(?:\s[A-Z][a-z]*)*\b'
    nomes_proprios = re.findall(regex, texto)

    return nomes_proprios
texto6 = """
Davi foi ao parque com Jonas e Bruno.
"""
nomes_proprios = extrair_nomes_proprios(texto6)
print(nomes_proprios)
# 10. **Contagem de Vogais:**
# Crie uma função que conte o número de vogais em uma string usando expressões regulares.
def contar_vogais(texto):
    regex = r'[aeiouAEIOU]'
    vogais = re.findall(regex, texto)
    numero_de_vogais = len(vogais)

    return numero_de_vogais
texto7 = "Maria é uma menina divertida."
numero_de_vogais = contar_vogais(texto7)
print(f"Número de vogais: {numero_de_vogais}")