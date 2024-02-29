import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('data.csv')

files = "agehigh.csv", "agemiddle.csv", "agelow.csv"

def makeGraph(data, color):
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    standard_scalar = StandardScaler()
    data = standard_scalar.fit_transform(data)

    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=1)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    xval = data[:, 0]
    yval = data[:, 1]

    plt.scatter(xval, yval, cmap='viridis',s=25, alpha=0.7, color=color)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=2, color='black',)
    plt.title("all_ages.csv")
    plt.legend()

files = "agehigh.csv", "agemiddle.csv", "agelow.csv"
colors = 'blue', 'green', 'red'
for (file, color) in zip(files,colors):
    data = pd.read_csv(file)
    makeGraph(data, color)

plt.show()