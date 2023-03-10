import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

iris = datasets.load_iris()
iris = pd.read_csv("Iris.csv")
iris.head()

# Clustering the iris dataset
iris['iris'].drop_duplicates()
iris_df = iris.drop('iris',axis=1)

# Instantiation
scaler = StandardScaler()

# Fit_transformation
iris_df_scaled = scaler.fit_transform(iris_df)
iris_df_scaled.shape

# The Elbow Method is the most effective technique for figuring out how many clusters the data may be divided into.
# Elbow Method: One of the most often used ways for figuring out this ideal value of k is the elbow approach.

# Implimenting the SSE
sse = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8 , 9 , 10 ]
for num_clusters in range_n_clusters:
kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
kmeans.fit(iris_df_scaled)

sse.append(kmeans.inertia_)

# plotting the number of clusters
plt.plot(ssd)

# Using the K-means
kmeans = KMeans(n_clusters=3, max_iter=50)
y = kmeans.fit_predict(iris_df_scaled)

# Giving the labels
iris_df['Label'] = kmeans.labels_
iris_df.head()

# Uisng K-means, visualising the clusters in iris data by scatter plot
plt.scatter(iris_df_scaled[y == 0, 0], iris_df_scaled[y == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(iris_df_scaled[y == 1, 0], iris_df_scaled[y == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(iris_df_scaled[y == 2, 0], iris_df_scaled[y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Other type of clustering is hierarchical clustering Complete Linkage
# The largest distance between any two points inside a cluster is used to establish the distance between two clusters in complete linkage hierarchical clustering.

# Implementing complete linkage
plt.figure(figsize=(15, 5))
mergings = linkage(iris_df_scaled,method='complete',metric='euclidean')
dendrogram(mergings)
plt.show()
