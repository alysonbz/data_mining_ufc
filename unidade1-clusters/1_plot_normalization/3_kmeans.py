import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import loadpokemon_dataset_df

# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

df= loadpokemon_dataset_df()

# Compute cluster centers
centroids,data = kmeans(df[['Attack', 'Defense']], num_clusters, inter=100)

# Assign cluster labels
df['cluster_labels'], _ = vq(df[['Attack', 'Defense']],centroids)

# Plot the points with seaborn
sns.__(x = "Attack", y = "Defense", hue="cluster_labels", data=df)
plt.show()