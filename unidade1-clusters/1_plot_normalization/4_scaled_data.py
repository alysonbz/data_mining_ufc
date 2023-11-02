import matplotlib.pyplot as plt
# Importe a função whiten
from scipy.cluster.vq import whiten

goals_for = [4, 3, 2, 3, 1, 1, 2, 0, 1, 4]

# Use a função whiten() para padronizar os dados
dados_padronizados = whiten(goals_for)
print(dados_padronizados)

# Plote os dados originais
plt.plot(goals_for, label='original')

# Plote os dados padronizados
plt.plot(dados_padronizados, label='padronizado')

# Mostre a legenda no gráfico
plt.legend()

# Exiba o gráfico
plt.show()
