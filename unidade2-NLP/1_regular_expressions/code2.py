# Importar os módulos necessários
from nltk.tokenize import sent_tokenize, word_tokenize
from src.utils import get_sample_Santo_Graal

# Dividir scene_one em sentenças: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Usar word_tokenize para tokenizar a quarta sentença: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Criar um conjunto de tokens únicos em toda a cena: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Imprimir o resultado dos tokens únicos
print(unique_tokens)
