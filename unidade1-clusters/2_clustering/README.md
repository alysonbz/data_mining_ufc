# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1

### Questão 1

[1_dendrogram.py](1_dendrogram.py)

#### Dendrograma

Nesta questão você visualizar o conjunto de dados via dendrograma

#### Instruções:

1)  Utilize a função linkagem para criar a matriz de distâncias com as colunas 'x_scaled' e 'y_scaled' do dataframe df. Utilize o método ward e distancia euclidiana.
   
2)  Chame e função ``dendrogram`` . Passe a matriz de distâncias como argumento 

3)  Mostre o dendrograma dos pontos com o método `` show`` da matplotlib .

### Questão 2

[2_Hierarchical_clustering.py](2_Hierarchical_clustering.py)

#### Clusterização Hierárquica

Nesta questão você vai aplicar uma clusterização hierárquica.

#### Instruções 

1) Importe o módulo ``linkage `` e ``fcluster``.
2) utilize a função linkage com o método ward.
3) Crie uma nova coluna no dataframe df chamada ``cluster_labels``. Adicione nessa coluna 
as labels dos clusters que serão calculados com a função ``fcluster`` passe para função os argumentos Z, que representa a matriz de distâncias, 
a quantadidade de cluster defina igual 2.
4) Utilizando o módulo ``seaborn`` realize um ``scatterplot``. Defina os arguentos ``x`` = x, ``y``=y e ``hue`` = cluster_labels e ``data`` = df.

### Questão 3

[2_elbow_method.py](2_elbow_method.py)

#### Enontrar a quantidad de clusters para o kmedias

Nesta questão você vai verificar limitações do kmedias.

#### Instruções 

1) Importe o módulo ``kmeans `` da scipy
2) itere sobre a quantidade de clusters e instancie o kmedias para cada itereção alterando o valor de k. Use as colunas ``x_scaled`` e ``y_scaled`` do dataframe ``comic_con``.
3) Empilhe na lista distortions os valores das somas dos erros (distortion)
4) crie um dataframe que armazene os valores de distortion para cada k.
5) Utilizando o módulo ``seaborn`` realize um ``lineplot``. Defina os arguentos ``x`` = num_cluster, ``y``= distortion  e ``hue`` = cluster_labels e ``data`` = elbow_plot.

