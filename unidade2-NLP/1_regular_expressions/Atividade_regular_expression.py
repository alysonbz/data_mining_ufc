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

print("resultado primeira questão: ",cont_word(text))



#2. *Validação de E-mail:*
# Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares.

def validar_email(endereco_email):
    regex = r'^[\w-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
    if re.fullmatch(regex, endereco_email):
        print("Email Válido")
    else:
        print("Email Inválido!")

email_teste = "exemplo123@dominio.com"
print(validar_email(email_teste))

#3. *Extração de Números de Telefone:*
# Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares.

def extrair_telefones(texto):
    regex = r'\(?\b\d{2}\)?\s?-?\d{4,5}-?\d{4}\b'
    numeros_telefone = re.findall(regex, texto)

    return numeros_telefone

texto2 = """
(11) 91234-0934
11-91834-7243
11912342983
(11) 3456-0909
+55 85 91234-6623
123457
32 0284
"""
telefones = extrair_telefones(texto2)
print(telefones)

# 4. *Substituição de Palavras:*
#Crie uma função que substitua todas as ocorrências de "gato" por "cachorro" em um texto usando expressões regulares.
def substituir_palavra(texto, original, substituto):
    regex = fr'\b{original}\b'
    texto_modificado = re.sub(regex, substituto, texto)

    return texto_modificado

texto3 = "era uma vez um gato chamado joao."
texto_modificado = substituir_palavra(texto3, "gato", "cachorro")
print(texto_modificado)

#5. *Extração de URLs:*
# Escreva um programa que extraia todas as URLs de um texto usando expressões regulares.
def extrair_urls(texto):
    regex = r'https?://(?:www\.)?\S+(?:/|(?=\s))'
    urls = re.findall(regex, texto)

    return urls

texto4 = """
https://www.exemplo123.com
http://exemplo123.org.br
Acesse: https://www.clickjogos.com.br/
"""
urls = extrair_urls(texto4)
print(urls)

#6. *Verificação de Senha Segura:*
#Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais.
def validar_senha(senha):
    regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%?&])[A-Za-z\d@$!%?&]{8,}$'
    if re.match(regex, senha):
        print("Senha válida")
    else:
        print("Senha inválida!")

senha1 = "Senha@123@@@"
senha2 = "senha142536"

print(validar_senha(senha1))
print(validar_senha(senha2))

# 7. *Extração de Palavras:*
# Escreva uma função que extraia todas as palavras de uma string usando expressões regulares.
def extrair_palavras(texto):
    regex = r'\b\w+\b'
    palavras = re.findall(regex, texto)

    return palavras

texto5 = "menina bonita do laço de fita"
palavras = extrair_palavras(texto5)
print(palavras)

# 8. *Validação de Data:*
# Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares.
def validar_data(data):
    regex = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/([0-9]{4})$'

    if re.match(regex, data):
        print("Data válida.")
    else:
        print("Data inválida!")

data1 = "06/07/2003"
data2 = "01/08/2023"

print(validar_data(data1))
print(validar_data(data2))

#9. *Extração de Nomes Próprios:*
# Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando expressões regulares.
def extrair_nomes_proprios(texto):
    regex = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
    nomes_proprios = re.findall(regex, texto)

    return nomes_proprios

texto6 = """
João é chato
"""
nomes_proprios = extrair_nomes_proprios(texto6)
print(nomes_proprios)

# 10. *Contagem de Vogais:*
# Crie uma função que conte o número de vogais em uma string usando expressões regulares.
def contar_vogais(texto):
    regex = r'[aeiouAEIOU]'
    vogais = re.findall(regex, texto)
    numero_de_vogais = len(vogais)

    return numero_de_vogais

texto7 = "neymar lindo."
numero_de_vogais = contar_vogais(texto7)
print(f"Número de vogais: {numero_de_vogais}")

