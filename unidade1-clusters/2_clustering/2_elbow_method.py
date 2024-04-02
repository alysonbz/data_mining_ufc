import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans  # Correção aqui
import pandas as pd

from src.utils import load_comic_con_dataset

comic_con = load_comic_con_dataset()
print(comic_con.columns)
distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion, _ = kmeans(comic_con[['scaled_x', 'scaled_y']], i)  # Correção aqui
    distortions.append(distortion)

# Create a DataFrame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()
