import itertools
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from src.utils import get_pre_process_wiki_articles

# Criar um Dicionário a partir dos artigos: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Criar um MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Obter o quinto documento no corpus: doc
doc = corpus[4]

# Ordenar o doc por frequência: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Imprimir as 5 principais palavras do documento juntamente com a contagem
for word_id, word_count in bow_doc[:5]:
    print("The token", dictionary.get(word_id), "appears", word_count, "times")

# Criar o defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Escolher uma chave entre 0 e 10 e mostrar a contagem com uma função print
key = 3
print("The key", key, "in defaultdict has count:", total_word_count[key], '\n')

# Criar uma lista ordenada a partir do defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Imprimir as 5 principais palavras em todos os documentos juntamente com a contagem
for word_id, count in sorted_word_count[:5]:
    print(dictionary.get(word_id), count)