import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# import the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
x = data[["Annual Income", "Spending Score"]]

# standardize the data
standardized = StandardScaler().fit_transform(x)

# define the number of clusters
k = 5

# apply the KMeans algorithm
km = KMeans(n_clusters=k, random_state=42).fit(standardized)

# get the centroid and label values
centroids = km.cluster_centers_
labels = km.labels_

print("Standardized Data:")
print(standardized)
print("\nCluster Labels:")
print(labels)

# set the size of the graph
plt.figure(figsize=(10, 6))

# plot the data points in each cluster
for i in range(k):
    cluster = standardized[np.where(labels == i)]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')

# plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, c="purple", label="Centroid")

# display the graph
plt.xlabel("Annual Income (standardized)")
plt.ylabel("Spending Score (standardized)")
plt.legend()
plt.title("K-Means Clustering of Customer Data")
plt.show()
