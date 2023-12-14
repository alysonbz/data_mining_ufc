import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------
def apagar_arquivos_no_diretorio(diretorio):
    # Certifique-se de que o caminho é um diretório válido
    if os.path.isdir(diretorio):
        # Obtém a lista de arquivos no diretório
        arquivos = os.listdir(diretorio)

        # Verifica se o diretório está vazio
        if not arquivos:
            print(f"O diretório {diretorio} está vazio. Nada a apagar.")
        else:
            # Itera sobre os arquivos e os apaga
            for arquivo in arquivos:
                caminho_completo = os.path.join(diretorio, arquivo)
                try:
                    if os.path.isfile(caminho_completo):
                        os.remove(caminho_completo)
                    elif os.path.isdir(caminho_completo):
                        os.rmdir(caminho_completo)
                except Exception as e:
                    print(f"Erro ao apagar {caminho_completo}: {e}")

            print(f"Todos os arquivos em {diretorio} foram apagados.")
    else:
        print(f"{diretorio} não é um diretório válido.")

# -------------------------------------------------------
def exibir_grid_imagens(diretorios, quantidade_por_diretorio=10):
    fig, axs = plt.subplots(len(diretorios), quantidade_por_diretorio, figsize=(15, 5 * len(diretorios)))
    fig.suptitle('Grid de Imagens', fontsize=16)

    for i, diretorio in enumerate(diretorios):
        # Listar arquivos no diretório
        arquivos_imagem = [f for f in os.listdir(diretorio) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Selecionar as primeiras 10 imagens
        imagens_selecionadas = arquivos_imagem[:quantidade_por_diretorio]

        for j, nome_arquivo in enumerate(imagens_selecionadas):
            # Caminho completo para a imagem
            caminho_imagem = os.path.join(diretorio, nome_arquivo)

            # Carregar a imagem e verificar se é válida
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                print(f"Erro: Não foi possível carregar a imagem {caminho_imagem}.")
                continue

            # Exibir a imagem no subplot correspondente
            axs[i, j].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
            axs[i, j].axis('off')
            axs[i, j].set_title(f'D{i + 1}, I{j + 1}')

    plt.show()
# -------------------------------------------------------
def ajuste_de_janela(imagem, min_intensity, max_intensity):
    # Ajustar os valores de intensidade para o intervalo desejado
    imagem_ajustada = np.clip(imagem, min_intensity, max_intensity)

    # Normalizar os valores para o intervalo 0-255 para exibição
    imagem_normalizada = cv2.normalize(imagem_ajustada, None, 0, 255, cv2.NORM_MINMAX)

    return imagem_normalizada.astype(np.uint8)

# -------------------------------------------------------
def preprocessamento_tomografia(diretorio_entrada, diretorio_saida, min_intensity=100, max_intensity=200):
    # Verificar se o diretório de saída existe, se não, criar
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    # Caso tenha algum arquivo no diretorio de saida apague
    apagar_arquivos_no_diretorio(diretorio_saida)

    # Listar arquivos no diretório de entrada
    arquivos_imagem = [f for f in os.listdir(diretorio_entrada) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Contador de arquivos no início
    total_arquivos_inicio = len(arquivos_imagem)

    for nome_arquivo in arquivos_imagem:
        # Caminho completo para a imagem de entrada
        caminho_entrada = os.path.join(diretorio_entrada, nome_arquivo)

        # Carregando a imagem
        imagem = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)

        # Verificando se a imagem é válida
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem {nome_arquivo}.")
            continue

        # Ajuste de janela na imagem
        imagem_ajustada = ajuste_de_janela(imagem, min_intensity, max_intensity)

        # Caminho completo para a imagem de saída
        caminho_saida = os.path.join(diretorio_saida, nome_arquivo)

        # Salvando a imagem recortada e destacada no diretório de saída
        cv2.imwrite(caminho_saida, imagem_ajustada)

    # Contador de arquivos no final
    total_arquivos_final = len(os.listdir(diretorio_saida))

    # Exibindo a contagem de arquivos
    print(f"Quantidade de arquivos no início: {total_arquivos_inicio}")
    print(f"Quantidade de arquivos salvos no diretório final: {total_arquivos_final}")
    print('\n')


# Exemplo de uso da função
diretorio_entrada_benignos = r'C:\Users\joaod\Documents\Semestres\2023_02\data_mining\data_mining_ufc\AV2\projeto 1\all_nods_benignos'
diretorio_entrada_malignos = r'C:\Users\joaod\Documents\Semestres\2023_02\data_mining\data_mining_ufc\AV2\projeto 1\all_nods_malignos'

diretorio_saida_benignos = r'C:\Users\joaod\Documents\Semestres\2023_02\data_mining\data_mining_ufc\AV2\projeto 1\data_silver\all_nods_benignos'
diretorio_saida_malignos = r'C:\Users\joaod\Documents\Semestres\2023_02\data_mining\data_mining_ufc\AV2\projeto 1\data_silver\all_nods_malignos'

preprocessamento_tomografia(diretorio_entrada_benignos, diretorio_saida_benignos)
preprocessamento_tomografia(diretorio_entrada_malignos, diretorio_saida_malignos)

diretorios_1 = [diretorio_entrada_malignos, diretorio_saida_malignos]
diretorios_2 = [diretorio_entrada_benignos, diretorio_saida_benignos]

exibir_grid_imagens(diretorios_1)
exibir_grid_imagens(diretorios_2)
