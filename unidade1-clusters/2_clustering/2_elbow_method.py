import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from src.utils import load_comic_con_dataset
import pandas as pd
comic_con = load_comic_con_dataset()

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(comic_con[['x_scaled', 'y_scaled']])
    distortions.append(kmeans.inertia_)

# Create a DataFrame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()
