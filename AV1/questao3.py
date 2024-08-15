'''importe a função que você implementou na questão 2 e gere dois conjuntos de atributos numéricos. O primeiro, baseado no DF e ourtro baseado no TFIDF.
Plote os 10 termos com maior TF-IDF e os 10 temos com maior DF. Lembre de implementar em forma de função para que possa ser importada em outra questão.
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from questao2 import texto_processado

'''função para gerar dois conjuntos numericos para DF e TD-IDF'''
def gerar_features(data, coluna_texto):
    data[coluna_texto] = data[coluna_texto].apply(lambda x: ' '.join(x))

    # Vetorização de contagem (Count Vectorizer)
    vetorizar_contagem = CountVectorizer(stop_words='english', max_features=5000, min_df=5)
    contar_X = vetorizar_contagem.fit_transform(data[coluna_texto])
    contar_data = pd.DataFrame(contar_X.toarray(), columns=vetorizar_contagem.get_feature_names_out())
    contar_data_sum = contar_data.sum().sort_values(ascending=False).reset_index()
    contar_data_sum.columns = ['Termo', 'DF']
    
    # Vetorização TF-IDF
    vetorizar_tfidf = TfidfVectorizer(stop_words='english', max_features=5000, min_df=5)
    tfidf_X = vetorizar_tfidf.fit_transform(data[coluna_texto])
    data_tfidf = pd.DataFrame(tfidf_X.toarray(), columns=vetorizar_tfidf.get_feature_names_out())
    media_tfidf = data_tfidf.mean().sort_values(ascending=False).reset_index()
    media_tfidf.columns = ['Termo', 'TF-IDF']
    
    # Retornar os vetores transformados junto com os sumários
    return contar_X, tfidf_X, contar_data_sum, media_tfidf



'''função para plotagem de grafico para os dois termos'''
def top_termos(contar_data_sum, media_tfidf):
    plt.figure(figsize=(14, 8))
    
    # Plotar os 10 termos com maior DF
    plt.subplot(2, 1, 1)
    contar_data_sum.head(10).plot(kind='barh', x='Termo', y='DF', legend=False, color='skyblue', ax=plt.gca())
    plt.title('10 Termos com Maior Document Frequency (DF)')
    plt.xlabel('Frequência DF')
    plt.ylabel('Termo')
    
    plt.subplot(2, 1, 2)
    media_tfidf.head(10).plot(kind='barh', x='Termo', y='TF-IDF', legend=False, color='lightcoral', ax=plt.gca())
    plt.title('10 Termos com Maior TF-IDF')
    plt.xlabel('TF-IDF')
    plt.ylabel('Termo')
    
    plt.tight_layout()
    plt.show()


'''usando com o dataset'''
if __name__ == "__main__":
    # Importar o dataset
    train = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')

    # Aplicação da função de pré-processamento
    train = texto_processado(train, ['OriginalTweet'])
    
    train = train.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])

    # Geração de atributos
    contar_X_train, tfidf_X_train, contar_data_sum_train, data_tfidf_train = gerar_features(train, 'OriginalTweet_tokens')

    # Plotar os top 10 termos
    top_termos(contar_data_sum_train, data_tfidf_train)