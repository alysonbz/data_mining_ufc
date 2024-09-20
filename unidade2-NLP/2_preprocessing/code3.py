# Importar Dictionary
from gensim.corpora import Dictionary
from src.utils import get_pre_process_wiki_articles

# Criar um Dicionário a partir dos artigos: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Selecionar o id para a palavra "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Usar computer_id com o dicionário para imprimir a palavra
print('The word', dictionary.get(computer_id), 'has index', computer_id, 'in the dictionary')

# Criar um MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Imprimir os primeiros 10 ids de palavras com suas contagens de frequência do quinto documento
print(corpus[4][:10])