# AVALIAÇÃO 1 
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

Giovana : questão 1 - https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset
outras questões - https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset

SHELDA DE SOUZA RAMOS :  questão 1 -  https://www.kaggle.com/datasets/rkiattisak/smart-watch-prices
outras questões - https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews

ANA LIVIA SOUSA DAVI TAVEIRA: questão 1 - https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
outras questões - https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset

ERICK RAMOS COUTINHO : questão 1 - https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset
outras questões - https://www.kaggle.com/datasets/jocelyndumlao/consumer-review-of-clothing-product

LUCIANA SOUSA MARTINS: questão 1-  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 
outras questões - https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification

MAVERICK ALEKYNE DE SOUSA RIBEIRO: questao 1 -  https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
outras questões - https://www.kaggle.com/datasets/lucaspoo/steam-reviews-international

 EMILY CAMELO MENDONCA: questao 1 - https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
outras questões - https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

 LUIS SAVIO GOMES ROSA: questão 1 - https://www.kaggle.com/datasets/prathamtripathi/drug-classification
outras questões - https://www.kaggle.com/datasets/gorororororo23/european-restaurant-reviews

PAULO HENRIQUE SANTOS MARQUES: qiuestão 1 - https://www.kaggle.com/datasets/whenamancodes/predict-diabities 
outras questões - https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification

ERYKA CARVALHO DA SILVA:  questão 1 - https://www.kaggle.com/datasets/mssmartypants/water-quality
outras questões - https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset


### Questão 1

```questao1.py```

#### Instruções:

Em uma atividade de casa, você implementou manualmente o algoritmo K-means para realizar uma analise não supervisionada.
Nesta avaliação você deve generalizar para receber uma estrutura de mais de duas dimensões e garantir que sua implementação manual realize as seguinte exigências:
a) Deve haver uma classe Kmeans, em que o contrutor ao inicializar, recebe a quantidade de centroides para o algoritmo e o dataframe.
b) O código deve retornar a label que cada ponto do conjunto de dados pertence, com base na quantidade de cluster definida.
c) A definição da quantidade de cluster não pode ser baseada na coluna target do datasset, você deve informar a quantidade de cluster utilizando algum método para encontrar esse parâmetro.

### Questão 2

```questao2.py```

#### Instruções 

Faça o download do dataset e realize os pré-processamentos adequados. Selecione as colunas que você acredita ser
adequdada de analisar, remova caracteres desnecessários, ajuste o conjunto e tokenize o conjunto, criando uma função para, inclusive
poder ser importada em outras questões. Plote nessa questão, as cinco primeiras listas de tokens geradas. 


### Questão 3

```questao3.py```

#### Instruções

importe a função que você implementou na questão 2 e gere dois conjuntos de atributos numéricos. O primeiro, baseado no DF e ourtro baseado no TFIDF.
Plote os 10 termos com maior TF-IDF e os 10 temos com maior DF. Lembre de implementar em forma de função para que possa ser importada em outra questão.

### Questão 4

```questao4.py```

#### Instruções

Utilizando as funções immplementadas nas questões anteriores, aplique uma classificação com o algoritmo apropriado, comparando
o desempenho com as duas formas de extração de atributos implementadas.

### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br** até 14/08/2024.

### Apresentação 14/08/2024

Criar uma apresentação de aproximadadamente 20 minutos que mostra seu entendimento sobre o problema
e resultados alcançados.


### Soma da Composição da nota:

#### Qualidade das atividades de casa: 2
#### Qualidade das atividades de sala: 1
#### Qualidade dos códigos: 2 
#### Qualidade do relatório: 3
#### Qualidade da apresentação: 2
