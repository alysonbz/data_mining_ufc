import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from questao2 import preprocess_text

df = pd.read_csv(r'C:\Users\joaod\OneDrive\Documentos\Semestre_2023.2\data_minning_ufc\AV1\Data\Consumer Review of Clothing Product\data_amazon.xlsx - Sheet1.csv')
df.dropna(inplace=True)
#preprocess_text = preprocess_text()


# Função para gerar conjuntos de atributos numéricos (DF e TF-IDF)
def generate_numeric_attributes(df):
    # Aplicar a função de pré-processamento e tokenização
    df['Combined_Text'] = df['Review'] + ' ' + df['Title']
    df['Tokens'] = df['Combined_Text'].apply(preprocess_text)

    # Converter os tokens de volta para texto
    df['Text'] = df['Tokens'].apply(lambda tokens: ' '.join(tokens))

    # Criar conjuntos de atributos numéricos usando CountVectorizer e TfidfVectorizer
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    count_matrix = count_vectorizer.fit_transform(df['Text'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Text'])

    # Obter nomes de recursos
    count_feature_names = count_vectorizer.get_feature_names_out()
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    return count_matrix, tfidf_matrix, count_feature_names, tfidf_feature_names

# Função para plotar os 10 termos com maior TF-IDF e DF
def plot_top_terms(matrix, feature_names, title):
    # Calcular somatório ao longo das linhas
    sums = matrix.sum(axis=0).A1  # Convertendo a matriz esparsa para um array

    # Obter os índices dos 10 maiores valores
    indices = sums.argsort()[-10:][::-1]

    # Obter os termos correspondentes
    terms = [feature_names[i] for i in indices]

    # Criar um gráfico de barras
    plt.bar(terms, sums[indices])
    plt.title(title)
    plt.xlabel('Termos')
    plt.ylabel('Valor')
    plt.xticks(rotation=45, ha='right')
    plt.show()



# Gerar conjuntos de atributos numéricos
count_matrix, tfidf_matrix, count_feature_names, tfidf_feature_names = generate_numeric_attributes(df)

# Plotar os 10 termos com maior TF-IDF
plot_top_terms(tfidf_matrix, tfidf_feature_names, 'Top 10 Termos com Maior TF-IDF')

# Plotar os 10 termos com maior DF
plot_top_terms(count_matrix, count_feature_names, 'Top 10 Termos com Maior DF')
