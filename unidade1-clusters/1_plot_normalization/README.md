# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1

### Questão 1

[1_cluster_plot.py](1_cluster_plot.py)

#### Plotando conjunto de dados

Nesta questão você vai realizar uma predição com KNN.

#### Instruções:

1)  Importe `` pyplot `` da matplotlib .
   
2)  Chame o método ``scatter``  através do objeto plt . 

3)  Mostre a distribuição dos pontos com o método `` show`` .

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


### Questão 2.1

[2_1_clusters_similarity.py](2_1_clusters_similarity.py)

#### Cálculo da similaridade de clusters

Nesta questão você vai realizar um cálculo para determinar a similaridade entre clusters.

#### Instruções

1) Com a função ``compute_single_linkage`` calcule a ligação simples.
2) Com a função ``compute_complete_linkage`` calcule a ligação completa
3) Com a função ``compute_average_linkage`` calcule a ligação média.
4) Com a função  ``compute_centroid_linkage`` calcule o método do centroide.
5) Com função ``compute_ward_linkage`` calcule o método de ward. 

dica: você pode implementar funções auxiliares além das descritas.
