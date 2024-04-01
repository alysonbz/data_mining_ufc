import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df
from scipy.cluster.hierarchy import linkage, fcluster

df= loadpokemon_dataset_df()

# Use the linkage() function to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x=df['x'], y=df['y'], hue='cluster_labels', data=df)
plt.show()

