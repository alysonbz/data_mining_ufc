'''Faça o download do dataset e realize os pré-processamentos adequados. Selecione as colunas que você acredita ser
adequdada de analisar, remova caracteres desnecessários, ajuste o conjunto e tokenize o conjunto, criando uma função para, inclusive
poder ser importada em outras questões. Plote nessa questão, as cinco primeiras listas de tokens geradas. '''

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset="datatattle/covid-19-nlp-text-classification", unzip=True)

