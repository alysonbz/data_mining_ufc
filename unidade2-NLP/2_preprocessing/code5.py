from gensim.corpora.dictionary import Dictionary
from src.utils import get_pre_process_wiki_articles
from gensim.models import TfidfModel

# Criar um Dicionário a partir dos artigos: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Criar um MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Obter o quinto documento no corpus: doc
doc = corpus[4]

# Criar um novo TfidfModel usando o corpus: tfidf
tfidf = TfidfModel(corpus)

# Calcular os pesos tfidf do doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Imprimir os primeiros cinco pesos
print(tfidf_weights[:5])

# Ordenar os pesos do maior para o menor: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Imprimir as 5 palavras com maior peso
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
