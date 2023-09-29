import re

#1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões regulares.

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex, text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print(cont_word(text))


# 2. **Validação de E-mail:**
#   Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares.

def valid_email(text):
    regex = r'^[a-zA-Z0-9._-]+@([a-z0-9]+)(\.[a-z]{2,3})+$'
    if re.match(regex, text):
        return 'email validado'
    else:
        return 'email inválido'

print(valid_email('aluno@ufc.br'))
print(valid_email('alunoufc.br'))
print(valid_email('aluno@ufcbr'))
print(valid_email('aluno@ufc.brasil'))


# 3. **Extração de Números de Telefone:**
#   Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares.

def number_resgate(text):
    regex = r'[+][0-9]{13}'
    return re.findall(regex, text)

print(number_resgate('O numero de telefone de fulano é +5585992754637, e o do fulano é tal +55 85 992754637'))

# 4. Substituição de Palavras:
# Crie uma função que substitua todas as ocorrências de "gato" por "cachorro"  em um texto usando expressões regulares.

def substituir(text, palavra_nova, palavra_troca):
    return print(re.sub(palavra_troca, palavra_nova, text))

textq = 'meu deus meu gato é lindo'
troca = 'gato'
nova = 'cachorro'

substituir(textq, nova, troca)

# 5. **Extração de URLs:**
#   Escreva um programa que extraia todas as URLs de um texto usando expressões regulares.

def urls_extractor(text):
    regex = r'https?://\S+|www\.\S+'
    return print(re.findall(regex, text))

texto = "Esta é uma string com uma URL: https://www.example.com e outra URL: http://www.example2.com"

urls_extractor(texto)

# 6. **Verificação de Senha Segura:**
#   Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se
#   tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais.

def senha_segura(senha):
    # Verifique se a senha tem pelo menos 8 caracteres de comprimento
    if len(senha) < 8:
        print('Senha não segura: comprimento mínimo não atendido.')
        return False

    # Verifique se a senha contém pelo menos uma letra maiúscula
    if not re.search(r'[A-Z]', senha):
        print('Senha não segura: falta letra maiúscula.')
        return False

    # Verifique se a senha contém pelo menos uma letra minúscula
    if not re.search(r'[a-z]', senha):
        print('Senha não segura: falta letra minúscula.')
        return False

    # Verifique se a senha contém pelo menos um dígito
    if not re.search(r'\d', senha):
        print('Senha não segura: falta dígito.')
        return False

    # Verifique se a senha contém pelo menos um caractere especial
    if not re.search(r'[!@#$%^&*()-=_+[\]{}|;:",.<>?]', senha):
        print('Senha não segura: falta caractere especial.')
        return False

    # Se a senha passar por todas as verificações, é considerada segura
    return print('Senha Segura!')

# Exemplos de uso:
senha1 = "Senha123!"
senha2 = "fraca"

senha_segura(senha1)
senha_segura(senha2)

# 7. **Extração de Palavras:**
#   Escreva uma função que extraia todas as palavras de uma string usando expressões regulares.

def extrair_palavras(text):
    regex = r'\w+'
    return print(re.findall(regex, text))

texto = "Esta é uma string de exemplo com algumas palavras."
extrair_palavras(texto)

# 8. **Validação de Data:**
#   Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares.

def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'
    if re.match(padrao, data):
        return print('Data válida')
    else:
        return print('Data Invalida')

# Exemplos de uso:
data1 = "15/09/2023"
data2 = "30/02/2023"
data3 = "2023/09/15"

validar_data(data1)
validar_data(data2)
validar_data(data3)

# 9. **Extração de Nomes Próprios:**
#   Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando
#   expressões regulares.

def extrair_nomes_proprios(texto):
    padrao = r'\b[A-Z][a-zA-Z]+\b'
    return print(re.findall(padrao, texto))

texto = "O Joao encontrou a Maria na rua. Eles foram ao cinema juntos."
extrair_nomes_proprios(texto)

# 10. **Contagem de Vogais:**
#   Crie uma função que conte o número de vogais em uma string usando expressões regulares.

def vogais(text):
    regex = r'[aeiouAEIOU]'
    vogais = re.findall(regex, text)
    return print(f'Este é o número de vogais no texto: {len(vogais)}')

texto = "Esta é uma frase de exemplo."

vogais(texto)