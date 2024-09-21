# Tutorial e Detalhes

Esse documento exibe detalhes dos scripts e como executá-los.

1) É necessário executar PRIMEIRO o script **leitura_dataset.py**, pois ele é responsável por estruturar o dataset
em X=imagens e y=labels, extrair atributos utilizando uma função criada no script **extracao_atributos.py** e salvar
os dados em CSV.
2) Após executar o primeiro script, o próximo script a ser executado é o **classificacao.py**, responsável por dividir em treino e teste,
aplicar o algorítmo de classificação e gerar as métricas.

Detalhes do que contém nas pastas:
1) Pasta Documents: plantas.csv e readme_tutorial.md
2) Pasta imagens: três pastas com as imagens das plantas utilizadas
3) Pasta questões: classificacao.py e leitura_dataset.py
4) Pasta utils: extracao_atributos.py