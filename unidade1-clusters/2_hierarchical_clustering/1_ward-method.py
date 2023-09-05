from src.utils import load_comic_con_dataset
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt

# função load_comic_con_dataset
comic_con = load_comic_con_dataset()

# função linkage() para calcular a matriz de ligação
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='ward', metric='euclidean')

# rótulos de cluster usando a função fcluster com base em um critério de número máximo de clusters
comic_con['cluster_labels'] = fcluster(distance_matrix, t=2, criterion='maxclust')

# clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.show()
