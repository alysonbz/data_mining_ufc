import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class KMeansCustom:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine the optimal number of clusters using the Silhouette method
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Plot the Silhouette scores
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method for Optimal k')
        plt.show()

        # Based on the plot, choose the number of clusters with the highest Silhouette score
        self.n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

        # Perform K-means clustering with the chosen number of clusters
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(X_scaled)

        return labels

# Load your dataset into a DataFrame (replace 'data.csv' with your file path)
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Remove rows where 'bmi' is defined as 'N/A' and other preprocessing
df = df[df['bmi'] != 'N/A']
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df.dropna(inplace=True)

# Select the relevant columns for clustering
X = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]

# Create an instance of KMeansCustom with an initial number of clusters
kmeans_custom = KMeansCustom(n_clusters=3)

# Perform K-means clustering and get the cluster labels
cluster_labels = kmeans_custom.fit(X)

# Add the cluster labels to the DataFrame
df['cluster'] = cluster_labels

# Now 'df' contains a new column 'cluster' indicating which cluster each data point belongs to
