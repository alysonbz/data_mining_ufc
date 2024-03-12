import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df
df= loadpokemon_dataset_df()


# Import linkage and fcluster functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, df, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x=____, y=____, hue=____, data=df)
plt.show()