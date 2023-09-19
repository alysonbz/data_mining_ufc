import re

# 1. **Contagem de Correspondências:**
#   Escreva um programa que conte quantas vezes a palavra "Python" aparece em uma determinada string usando expressões
#   regulares.

def cont_word(text):
    regex = "Python"
    resp = re.findall(regex,text)
    return len(resp)

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

#print(cont_word(text))


# 2. **Validação de E-mail:**
#   Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares.
def valid_email(email):
    padrao = r'[a-zA-Z0-9._]+@[a-zA-Z0-9.]+\.[a-zA-Z]'
    if re.match(padrao, email):
        return 'email válido!'
    else:
        return 'email inválido!'


print(valid_email('abc@alu.ufc.br'))
print(valid_email('def@gmail.com'))
print(valid_email('ghi@gmailcom'))


# 3. **Extração de Números de Telefone:**
#   Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares.

def extrair_telefone(numero):
    padrao = r'\(?\d{2}\)?[-\s]?\d{9}'
    return re.findall(padrao, numero)

print(extrair_telefone('Meu número é (85) 991929394. Também pode ser escrito como 85-991929394 ou 85 991929394.'))


# 4. Substituição de Palavras:
# Crie uma função que substitua todas as ocorrências de "gato" por "cachorro"  em um texto usando expressões regulares.

def substituir_palavra(texto, p_antiga, p_nova):
    return re.sub(p_antiga, p_nova, texto)

texto = 'eu tenho um gato.'
p_antiga = 'gato'
p_nova = 'cachorro'

print(substituir_palavra(texto, p_antiga, p_nova))


# 5. **Extração de URLs:**
#   Escreva um programa que extraia todas as URLs de um texto usando expressões regulares.

def extrair_urls(texto):
    padrao = r'https?://\S+|www\.\S+'
    return re.findall(padrao, texto)

texto = 'O url do portal é https://si3.ufc.br/sigaa. O da biblioteca é https://pergamum.ufc.br/pergamum/biblioteca/index.php'

print(extrair_urls(texto))


# 6. **Verificação de Senha Segura:**
#   Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se
#   tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais.


# 7. **Extração de Palavras:**
#   Escreva uma função que extraia todas as palavras de uma string usando expressões regulares.


# 8. **Validação de Data:**
#   Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares.


# 9. **Extração de Nomes Próprios:**
#   Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando
#   expressões regulares.


# 10. **Contagem de Vogais:**
#   Crie uma função que conte o número de vogais em uma string usando expressões regulares.


