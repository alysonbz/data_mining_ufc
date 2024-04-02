from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from src.utils import load_fifa_dataset

from scipy.cluster.hierarchy import linkage, fcluster

fifa = load_fifa_dataset()
print(fifa.columns)


# Fit the data into a k-means algorithm
cluster_centers,_ = kmeans(fifa[['sliding_tackle', 'aggression']])

# Assign cluster labels
fifa['cluster_labels'], _ = fcluster(cluster_centers, 2, criterion='max_clust')

# Display cluster centers

print(fifa.groupby('cluster_labels')[['sliding_tackle', 'aggression']].mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='sliding_tackle', y='aggression', hue='cluster_labels', data=fifa)
plt.title('K-Means Clustering of FIFA Dataset')
plt.xlabel('Sliding Tackle')
plt.ylabel('Aggression')
plt.show()
