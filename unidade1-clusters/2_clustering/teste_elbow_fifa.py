import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#Import kmeans from scipy
from scipy.cluster.vq import kmeans

from src.utils import load_fifa_dataset

fifa = load_fifa_dataset()

distorcions = []
num_of_clusters = range(2,10)

for i in num_of_clusters:
    centroids, distorcion = kmeans(fifa[['sliding_tackle','aggression']],i)
    distorcions.append(distorcion)

elbow_plot = pd.DataFrame({'num_clusters': num_of_clusters, 'distortions': distorcions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x="num_clusters", y="distortions", data = elbow_plot)
plt.xticks(num_of_clusters)
plt.show()
