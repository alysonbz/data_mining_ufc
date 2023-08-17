from src.utils import load_comic_con_dataset
from scipy.cluster.hierarchy import ____, ____

comic_con = load_comic_con_dataset()

# Import the fcluster and linkage functions

# Use the linkage() function
distance_matrix = ____(comic_con[['x_scaled', 'y_scaled']], ____ = ____, metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = ____(____, ____, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data = comic_con)
plt.show()