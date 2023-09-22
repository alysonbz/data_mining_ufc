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

print(cont_word(text))
#2 valide um email
def validacao(email):
    validar = r"[^@]+@\S+\.[^@]+"
    if re.match(validar,email):
        print('esse eh um email valido')
    else:
        print('nao eh um email valido')

email = "mateus@email.com"
validacao(email)
#3 escreva um programa que pegue todos os numeros telefonicos de um texto
def cont(text):
    regex = r'[0-9]+'
    resp = re.findall(regex,text)
    return resp
text = 'o numero eh 85992773007,'\
        'o numero da seu amigo 85992415636,'\
        'o numero da sua irma 85992745620'
numeros_telefonicos = cont(text)

for numero in numeros_telefonicos:
    print(numero)
#4 substituição da palavra gato por cachorro
def mude(text):
    resp = re.sub(r'gato','cachorro',text)
    return resp
text = 'mateus possui um gato,'\
        'iarley comprou um gato,'\
        'levi perdeu seu gato'
print(mude(text))

#5 escreva um programa que encontre urls em um texto
def ache_urls(text):
    procure_url = r'https:\/\/[^\s/$.?#].[^\s]*'
    urls = re.findall(procure_url, text)
    return urls

text = 'mateus visitou o site de notícias em https://www.example-news.com para se manter atualizado,'\
        ' levi encontrou um tutorial útil em https://www.example-tutorial.com para aprender a programar.'\
        ' iarley compartilhou fotos incríveis em seu álbum do Instagram em https://www.instagram.com/carla_photos.'

urls_encontradas = ache_urls(text)

for url in urls_encontradas:
    print(url)
# 6 verifique se uma senha é segura

def verificar_senha(senha):
    r = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*?])[A-Za-z\d!@#$%^&*?]{8,}$')
    if len(senha) >= 8 and r.match(senha):
        return "Senha válida"
    else:
        return "Senha inválida"
senhas = ['mateus', 'Mateus1@', 'mateus1', 'mmm@']
for senha in senhas:
    resultado = verificar_senha(senha)
    print(f'Senha: {senha} - {resultado}')
#7 funçao que pega todas as palavras de uma string
def cont_word(text):
    regex = r'\b\w+\b'
    resp = re.findall(regex,text)
    return resp

text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."

print(cont_word(text))

# 8 valide data


def validar_data(data):
    regex = re.compile(r"(?P<dia>\d{2})/(?P<mes>\d{2})/(?P<ano>20\d{2})")
    match = regex.match(data)

    if match:
        dia = int(match.group('dia'))
        mes = int(match.group('mes'))
        ano = int(match.group('ano'))
        if mes >= 1 and mes <= 12:
            if mes in [1, 3, 5, 7, 8, 10, 12]:
                max_dia = 31
            elif mes == 2:
                if (ano % 4 == 0 and ano % 100 != 0) or (ano % 400 == 0):
                    max_dia = 29
                else:
                    max_dia = 28
            else:
                max_dia = 30
            if dia >= 1 and dia <= max_dia:
                return True

    return False
data = "21/11/2002"
if validar_data(data):
    print(f"A data {data} é válida.")
else:
    print(f"A data {data} é inválida.")
# 9 nome proprio

def nome_proprio(text):
    regex = r"[A-Z]\w+"
    resp = re.findall(regex,text)
    return resp
text = "Mateus eh lindo,"\
        "Iarley eh legal"
print(nome_proprio(text))
def vogais(text):
    regex =  r'[aeiouAEIOU]'
    resp = re.findall(regex,text)
    return len(resp)
text = "A liguagem de programação em Python é a mais utilizada do mundo," \
       " fazendo de Python também a principal ferramenta para pesquisa de ML. " \
       "Ou seja, python é presente e futuro."
print(vogais(text))