"""
single linkage = Definido pela menor distancia de qualquer ponto do primeiro cluster
com algum elemento do segundo cluster.
"""

"""
complete Linkage = definida pela maior distancia de qualquer ponto do primeiro cluster
com segundo cluster.
"""

"""
Avarage Linkage = É definida pela média das distancias de todos os pontos do primeiro cluster 
com relação ao segundo cluster.
"""

"""
Centrouid Method = medida de similaridade definida pelo ponto médio do primeiro e segundo cluster.
"""

"""
wards method ou método da mínima variancia = Medida de dictancia etre dois clusters. É a soma das
distancias ao quadrado entre os dois clusters.
"""

"""
A função fcusters adiciona rotulo aos dados. Se for passado o parâmetro maxclus, siginifica
que sera realizado uma análise de algoritimo e se existir mais clusters que o parâmetro máximo
informado, o algoritimo combinara os clusters menos semelhiantes até que fique com a quantidade
de clusters passados no parâmetro.
"""

