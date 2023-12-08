#Bibliotecas usadas para o projeto
import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
#carrega os arquivos
#as imagens não possuam a mesma dimensão então tive que ajustar
def carregar_e_redimensionar_imagens(diretorio, tamanho_alvo):
    caminhos_imagens = [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio)]
    imagens = [cv2.resize(cv2.imread(caminho_imagem), tamanho_alvo) for caminho_imagem in caminhos_imagens]
    return imagens
#para facilitar a analise teve o processo de normalização
def normalizar_imagem(imagem):
    imagem_normalizada = cv2.normalize(imagem, None, 0, 255, cv2.NORM_MINMAX)
    return imagem_normalizada
#correção da atmosfera é uma etapa importante na analise de satelite
def corrigir_atmosfera(imagem, ganho=2.0, gamma=1.0):
    img_array = np.array(imagem, dtype=float)
    img_array = img_array * ganho
    img_array = np.power(img_array, gamma)
    return img_array
def calcular_entropia(imagem):
    imagem_gray = cv2.cvtColor(imagem.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    entropia = scipy.stats.entropy(imagem_gray.flatten())
    return entropia
#usei o calculo do ndvi para classificação
def calcular_ndvi(imagem):
    imagem = imagem.astype(float)
    red = imagem[:, :, 0]
    blue = imagem[:, :, 2]
    divisor = red + blue
    divisor[divisor == 0] = 1
    ndvi = (red - blue) / divisor
    return ndvi
#parte de extração de atributos
def extrair_atributos(imagens):
    atributos = []
    for imagem in imagens:
        imagem_normalizada = normalizar_imagem(imagem)
        imagem_corrigida = corrigir_atmosfera(imagem_normalizada, ganho=2.0, gamma=1.0)
        ndvi = calcular_ndvi(imagem_corrigida)
        entropia = calcular_entropia(imagem_corrigida)
        atributos.append([np.mean(ndvi), np.std(ndvi), entropia])  # Adicione mais índices conforme necessário
    return atributos

# Diretórios para cada classe
diretorios_classes = [
    r"C:/Users/mateu/Downloads/projetos/projetos/projeto 6/data/water",
    r"C:/Users/mateu/Downloads/projetos/projetos/projeto 6/data/green_area",
    r"C:/Users/mateu/Downloads/projetos/projetos/projeto 6/data/desert",
    r"C:/Users/mateu/Downloads/projetos/projetos/projeto 6/data/cloudy"
]

# Carregar e redimensionar imagens
imagens_agua = carregar_e_redimensionar_imagens(diretorios_classes[0], (150, 150))
imagens_area_verde = carregar_e_redimensionar_imagens(diretorios_classes[1], (150, 150))
imagens_deserto = carregar_e_redimensionar_imagens(diretorios_classes[2], (150, 150))
imagens_nublado = carregar_e_redimensionar_imagens(diretorios_classes[3], (150, 150))

# Extração de atributos
atributos_agua = extrair_atributos(imagens_agua)
atributos_area_verde = extrair_atributos(imagens_area_verde)
atributos_deserto = extrair_atributos(imagens_deserto)
atributos_nublado = extrair_atributos(imagens_nublado)

# Criação do DataFrame
df_agua = pd.DataFrame({'NDVI_Media': [x[0] for x in atributos_agua],
                        'NDVI_Desvio_Padrao': [x[1] for x in atributos_agua],
                        'Classe': 'Agua'})

df_area_verde = pd.DataFrame({'NDVI_Media': [x[0] for x in atributos_area_verde],
                              'NDVI_Desvio_Padrao': [x[1] for x in atributos_area_verde],
                              'Classe': 'Area Verde'})

df_deserto = pd.DataFrame({'NDVI_Media': [x[0] for x in atributos_deserto],
                           'NDVI_Desvio_Padrao': [x[1] for x in atributos_deserto],
                           'Classe': 'Deserto'})

df_nublado = pd.DataFrame({'NDVI_Media': [x[0] for x in atributos_nublado],
                           'NDVI_Desvio_Padrao': [x[1] for x in atributos_nublado],
                           'Classe': 'Nublado'})
df_agua['Entropia'] = [calcular_entropia(imagem) for imagem in imagens_agua]
df_area_verde['Entropia'] = [calcular_entropia(imagem) for imagem in imagens_area_verde]
df_deserto['Entropia'] = [calcular_entropia(imagem) for imagem in imagens_deserto]
df_nublado['Entropia'] = [calcular_entropia(imagem) for imagem in imagens_nublado]


# Concatenação dos DataFrames
df = pd.concat([df_agua, df_area_verde, df_deserto, df_nublado], ignore_index=True)
#essa parte para analise se está correndo bem o codigo
def imprimir_valores_atributos(atributos, classe):
    # Imprime os valores médios e desvios padrão para os 5 primeiros
    medias = [round(np.mean(valor), 2) for valor in atributos[:5]]
    desvios_padrao = [round(np.std(valor), 2) for valor in atributos[:5]]

    print(f"Valores de atributos para a classe {classe} (apenas os 5 primeiros):\n")
    print(f"Médias: {medias}\n")
    print(f"Desvios Padrão: {desvios_padrao}\n")

# Chama a função para imprimir os valores das classes
imprimir_valores_atributos(atributos_agua, 'Água')
imprimir_valores_atributos(atributos_area_verde, 'Área Verde')
imprimir_valores_atributos(atributos_deserto, 'Deserto')
imprimir_valores_atributos(atributos_nublado, 'Nublado')


def imprimir_primeiras_cinco_imagens(imagens, classe):
    for i in range(min(5, len(imagens))):  # Garante que imprimimos no máximo 5 imagens
        imagem_normalizada = normalizar_imagem(imagens[i])
        imagem_corrigida = corrigir_atmosfera(imagem_normalizada, ganho=2.0, gamma=1.0)
        imagem_corrigida_gray = cv2.cvtColor(imagem_corrigida.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Imprime a imagem original
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(imagens[i], cv2.COLOR_BGR2RGB))
        plt.title(f"Original {i + 1}")
        plt.axis('off')

        # Imprime a imagem corrigida
        plt.subplot(2, 5, i + 6)
        plt.imshow(imagem_corrigida.astype(np.uint8), cmap='gray')
        plt.title(f"Corrigida {i + 1}")
        plt.axis('off')

    plt.suptitle(classe, y=1.02)  # Adiciona título geral acima dos subplots
    plt.show()

# Chama a função para imprimir as imagens das classes
imprimir_primeiras_cinco_imagens(imagens_agua, 'Água')
imprimir_primeiras_cinco_imagens(imagens_area_verde, 'Área Verde')
imprimir_primeiras_cinco_imagens(imagens_deserto, 'Deserto')
imprimir_primeiras_cinco_imagens(imagens_nublado, 'Nublado')
# ... (seu código anterior)

# Análise de Distribuição das Classes
def analisar_distribuicao_classes(dataframe):
    distribuicao = dataframe['Classe'].value_counts(normalize=True) * 100

    # Imprima os resultados
    print("Análise de Distribuição das Classes:\n")
    for classe, percentual in distribuicao.items():
        print(f"Classe {classe}: Média de {percentual:.2f}% das amostras.")

    # Plotar gráfico de barras
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Classe', data=dataframe, palette='bright')
    plt.title('Distribuição das Classes')
    plt.xlabel('Classe')
    plt.ylabel('Contagem')
    plt.show()

# Chama a função para analisar a distribuição das classes
analisar_distribuicao_classes(df)


# Separar em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df[['NDVI_Media', 'NDVI_Desvio_Padrao']], df['Classe'], test_size=0.2, random_state=30)

# Lidar com desequilíbrio de classe usando RandomOverSampler
ros = RandomOverSampler(random_state=30)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Verificar a distribuição das classes após o balanceamento
for classe in df['Classe'].unique():
    percentual = (y_resampled == classe).sum() / len(y_resampled) * 100
    print(f'Classe {classe}: Média depois do balanceamento: {percentual:.2f}% das amostras.')

# Use um classificador MLP com ajuste de hiperparâmetros
modelo_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=25, solver='adam')
modelo_mlp.fit(X_resampled, y_resampled)

# Predição no conjunto de teste
y_pred_mlp = modelo_mlp.predict(X_test)

# Métricas de desempenho para MLP
acuracia_mlp = accuracy_score(y_test, y_pred_mlp)
relatorio_classificacao_mlp = classification_report(y_test, y_pred_mlp, zero_division=1)
matriz_confusao_mlp = confusion_matrix(y_test, y_pred_mlp)

print(f'Acurácia do modelo MLP: {acuracia_mlp:.4f}\n')
print('Relatório de Classificação (MLP):\n', relatorio_classificacao_mlp)
print('Matriz de Confusão (MLP):\n', matriz_confusao_mlp)


# Plotar a matriz de confusão para MLP
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_mlp, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=modelo_mlp.classes_, yticklabels=modelo_mlp.classes_)
plt.title('Matriz de Confusão - MLP')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()
#Usando Modelo SVM
modelo_svm = SVC(kernel='linear', random_state=25)
modelo_svm.fit(X_resampled, y_resampled)

# Predição no conjunto de teste
y_pred_svm = modelo_svm.predict(X_test)

# Métricas de desempenho para SVM
acuracia_svm = accuracy_score(y_test, y_pred_svm)
relatorio_classificacao_svm = classification_report(y_test, y_pred_svm, zero_division=1)
matriz_confusao_svm = confusion_matrix(y_test, y_pred_svm)

print(f'Acurácia do modelo SVM: {acuracia_svm:.4f}\n')
print('Relatório de Classificação (SVM):\n', relatorio_classificacao_svm)
print('Matriz de Confusão (SVM):\n', matriz_confusao_svm)

# Plotar a matriz de confusão para SVM
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_svm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=modelo_svm.classes_, yticklabels=modelo_svm.classes_)
plt.title('Matriz de Confusão - SVM')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# Classificador KNN
modelo_knn = KNeighborsClassifier(n_neighbors=8)
modelo_knn.fit(X_resampled, y_resampled)

# Predição no conjunto de teste
y_pred_knn = modelo_knn.predict(X_test)

# Métricas de desempenho para KNN
acuracia_knn = accuracy_score(y_test, y_pred_knn)
relatorio_classificacao_knn = classification_report(y_test, y_pred_knn, zero_division=1)
matriz_confusao_knn = confusion_matrix(y_test, y_pred_knn)

print(f'Acurácia do modelo KNN: {acuracia_knn:.4f}\n')
print('Relatório de Classificação (KNN):\n', relatorio_classificacao_knn)
print('Matriz de Confusão (KNN):\n', matriz_confusao_knn)

# Plotar a matriz de confusão para KNN
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=modelo_knn.classes_, yticklabels=modelo_knn.classes_)
plt.title('Matriz de Confusão - KNN')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()


df.to_csv('atributos_dataset_atualizado.csv', index=False)
