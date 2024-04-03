import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df

# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

df= loadpokemon_dataset_df()
print(df.columns)

# Compute cluster centers
centroids,data = kmeans(df, 3)

# Assign cluster labels
df['cluster_labels'], _ = vq(df,centroids)

# Plot the points with seaborn
sns.scatterplot(x = 'x', y = 'y', hue='cluster_labels', data=df)
plt.show()