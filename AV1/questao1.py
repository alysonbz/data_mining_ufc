import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'C:\Users\mateu\Downloads\archive (1)\heart.csv'
df = pd.read_csv(file_path)
def age_fase(age):
    age_fase = []
    for a in age:
        if a<45:
            age_fase.append('adult')
        elif a<60:
            age_fase.append('midle-age')
        else:
            age_fase.append('old')
    return pd.Series(age_fase)
def BP_level(BP):
    PBP_level = []
    for bp in BP:
        if bp<120:
            PBP_level.append('normal')
        elif bp<130:
            PBP_level.append('elevated')
        elif bp<140:
            PBP_level.append('hypert_s1')   #Hypertension stage 1
        elif bp<180:
            PBP_level.append('hypert_s2')
        else:
            PBP_level.append('hypert_crisis')   #Hypertension crise
    return pd.Series(PBP_level)
def Cholesterol_level(cholesterol):
    clt_level = []
    for c in cholesterol:
        if c<200:
            clt_level.append('healthy')
        elif c<240:
            clt_level.append('borderline')
        else:
            clt_level.append('dangerous')
    return pd.Series(clt_level)
def FastingBS_level(bs):
    FastBS = []
    for b in bs:
        if b==1:
            FastBS.append('diabetic')
        else:
            FastBS.append('normal')
    return pd.Series(FastBS)
def FastingBS_level(bs):
    FastBS = []
    for b in bs:
        if b==1:
            FastBS.append('diabetic')
        else:
            FastBS.append('normal')
    return pd.Series(FastBS)
def level_heart_rate(age,max_heart):
    level_heart = []
    for a,h in zip(age,max_heart):
        max_rate = (220-a)*0.8
        min_rate = (220-a)*0.5
        if h<min_rate:
            level_heart.append('low')
        elif h<=max_rate:
            level_heart.append('ideal')
        else:
            level_heart.append('high')
    return pd.Series(level_heart)
df['MaxHR'] = level_heart_rate(df['Age'],df['MaxHR'])
df['Age'] = age_fase(df['Age'])
df['RestingBP'] = BP_level(df['RestingBP'])
df['Cholesterol'] = Cholesterol_level(df['Cholesterol'])
df['FastingBS'] = FastingBS_level(df['FastingBS'])
df_d = pd.get_dummies(df)
df_d.head()
x = df_d.div(df_d.sum(axis=1),axis='rows')
x.head()

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        self.centroids = X[random.sample(range(X.shape[0]), self.n_clusters)]

        for _ in range(self.max_iter):
            labels = self.assign_clusters(X)
            new_centroids = self.compute_centroids(X, labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

        return labels

    def assign_clusters(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            distances = np.sum((point - self.centroids) ** 2, axis=1)
            labels[i] = np.argmin(distances)
        return labels

    def compute_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(X[labels == i], axis=0)
        return new_centroids

X = x.values



distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit(X)
    centroid_dist = np.sum((X - kmeans.centroids[labels]) ** 2)
    distortions.append(centroid_dist)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Distorção')
plt.title('Método Elbow para Determinação do Número de Clusters')
plt.show()

num_clusters = int(input("Digite o número de clusters com base no gráfico Elbow: "))
kmeans = KMeans(n_clusters=num_clusters)
labels = kmeans.fit(X)
df['cluster'] = labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.s
catter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X')
plt.show()



