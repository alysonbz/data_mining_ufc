import re


# 1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões
#   regulares.

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex, text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print('Questão 1 ------------------------')
print(cont_word(text))


# 2. **Validação de E-mail:**
#   Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares.
def valid_email(email):
    padrao = r'[\w+\.]+@[\w+\.]+\.[a-zA-Z]'
    if re.match(padrao, email):
        return 'email válido!'
    else:
        return 'email inválido!'

print('\nQuestão 2 ------------------------')
print('Email: abc@alu.ufc.br\n', valid_email('abc@alu.ufc.br'))
print('Email: def@gmail.com\n', valid_email('def@gmail.com'))
print('Email: ghi@gmailcom\n', valid_email('ghi@gmailcom'))


# 3. **Extração de Números de Telefone:**
#   Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares.

def extrair_telefone(numero):
    padrao = r'\(?\d{2}\)?[-\s]?\d{9}'
    return re.findall(padrao, numero)

print('\nQuestão 3 ------------------------')
texto = 'Meu número é (85) 991929394. Também pode ser escrito como 85-991929394 ou 85 991929394.'
print('Texto:', texto)
print(extrair_telefone(texto))


# 4. Substituição de Palavras:
# Crie uma função que substitua todas as ocorrências de "gato" por "cachorro"  em um texto usando expressões regulares.

def substituir_palavra(texto, p_antiga, p_nova):
    return re.sub(p_antiga, p_nova, texto)

texto = 'eu tenho um gato.'
p_antiga = 'gato'
p_nova = 'cachorro'

print('\nQuestão 4 ------------------------')
print('texto:', texto)
print(substituir_palavra(texto, p_antiga, p_nova))


# 5. **Extração de URLs:**
#   Escreva um programa que extraia todas as URLs de um texto usando expressões regulares.

# padrao = urls que comecem com https:// ou www.
# /S+ = qualquer caractere, sem espaços.
def extrair_urls(texto):
    padrao = r'https://\S+|www.\S+'
    return re.findall(padrao, texto)

texto = 'O url do portal é https://si3.ufc.br/sigaa. O do YouTube é www.youtube.com.'

print('\nQuestão 5 ------------------------')
print('Texto:', texto)
print(extrair_urls(texto))


# 6. **Verificação de Senha Segura:**
#   Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se
#   tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais.

# ?=.* PELO MENOS 1
def senha_segura(senha):
    padrao = r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@#$%&+=!]).{8,}$'
    if re.match(padrao, senha):
        return 'senha válida!'
    else:
        return 'senha inválida!'


print('\nQuestão 6 ------------------------')
print('Senha: Senha@123\n', senha_segura('Senha@123'))
print('Senha: senhafraca\n', senha_segura('senhafraca'))
print('Senha: Senha123\n', senha_segura('Senha123'))


# 7. **Extração de Palavras:**
#   Escreva uma função que extraia todas as palavras de uma string usando expressões regulares.
def extrair_palavras(texto):
    padrao = r'\b\w+\b'
    palavras = re.findall(padrao, texto)
    return palavras


texto = 'A liguagem de programação em Python é a mais utilizada do mundo,' \
        ' fazendo de Python também a principal ferramenta para pesquisa de ML. ' \
        'Ou seja, python é presente e futuro.'
palavras = extrair_palavras(texto)
print('\nQuestão 7 ------------------------')
print('Texto:', texto)
print(palavras)


# 8. **Validação de Data:**
#   Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares.

# (0[1-9]|[12][0-9]|3[01]) = 31 dias
# (0[1-9]|1[0-2]) - 12 meses
# \d{4} - yyyy
def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'
    if re.match(padrao, data):
        return 'data válida!'
    else:
        return 'data inválida!'


print('\nQuestão 8 ------------------------')
print('Data: 20/09/2023\n', validar_data('20/09/2023'))
print('Data: 38/12/2023\n', validar_data('38/12/2023'))


# 9. **Extração de Nomes Próprios:**
#   Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando
#   expressões regulares.

# '*' zero ou mais
def extrair_nomes_proprios(texto):
    padrao = r'\b[A-Z][a-z]*\b'
    nomes_proprios = re.findall(padrao, texto)
    return nomes_proprios


texto = 'A liguagem de programação em Python é a mais utilizada do mundo,' \
        ' fazendo de Python também a principal ferramenta para pesquisa de ML. ' \
        'Ou seja, python é presente e futuro.'
print('\nQuestão 9 ------------------------')
print('Texto:', texto)
nomes = extrair_nomes_proprios(texto)
print(nomes)

# 10. **Contagem de Vogais:**
#   Crie uma função que conte o número de vogais em uma string usando expressões regulares.

def contar_vogais(texto):
    padrao = r'[aeiouAEIOU]'
    vogais = re.findall(padrao, texto)
    return len(vogais)


texto = 'A liguagem de programação em Python é a mais utilizada do mundo,' \
        ' fazendo de Python também a principal ferramenta para pesquisa de ML. ' \
        'Ou seja, python é presente e futuro.'
print('\nQuestão 10 ------------------------')
print('Texto:', texto)
numero_de_vogais = contar_vogais(texto)
print(numero_de_vogais)
