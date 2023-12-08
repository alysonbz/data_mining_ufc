from processamento_imagem import normalizar_imagem,corrigir_atmosfera,calcular_ndvi,carregar_e_redimensionar_imagens,modelo_mlp,diretorios_classes
import numpy as np
def extrair_atributos_teste(imagens_teste):
    atributos_teste = []
    for imagem_teste in imagens_teste:
        imagem_normalizada_teste = normalizar_imagem(imagem_teste)
        imagem_corrigida_teste = corrigir_atmosfera(imagem_normalizada_teste, ganho=2.0, gamma=1.0)
        ndvi_teste = calcular_ndvi(imagem_corrigida_teste)
        atributos_teste.append([np.mean(ndvi_teste), np.std(ndvi_teste)])  # Adicione mais índices conforme necessário
    return atributos_teste
def classificar_novos_dados(modelo, atributos_novos):
    return modelo.predict(atributos_novos)
# Exemplo de Mineração de Atributos no Conjunto de Teste
imagens_teste_agua = carregar_e_redimensionar_imagens(diretorios_classes[0] + "_teste", (150, 150))
atributos_teste_agua = extrair_atributos_teste(imagens_teste_agua)

# Exemplo de Classificação de Novos Dados
resultado_predicao_agua = classificar_novos_dados(modelo_mlp, atributos_teste_agua)
print("Resultado da Classificação para Novas Imagens de Água:", resultado_predicao_agua)
#essa parte foi utilizada para possiveis novas imagens, porem testei 100%