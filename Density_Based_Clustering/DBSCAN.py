import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from pylab import *
import matplotlib.pyplot as plt

#create exampledataset
centers = [[1,1], [-1,-1], [1, -1]]
X, labels_true = make_blobs(n_samples=800, centers=centers, cluster_std=0.4, random_state=0)

#scale and standardize data
X = StandardScaler().fit_transform(X)

# #show the data
# xx, yy = zip(*X)
# plt.scatter(xx,yy)
# plt.show()

#define DBSCAN-parameters
db = DBSCAN(eps=0.2, min_samples=10).fit(X)
core_samples = db.core_sample_indices_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

#amount of clusters
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) 

#if the label is '-1' the datapoint is an outlier
outliers = X[labels == -1]

#separate the clusters
cluster_1 = X[labels == 1]
cluster_2 = X[labels == 2]
cluster_3 = X[labels == 3]

#plotting the clusters and outliers
unique_labels = set(labels)
colors = ['b', 'g', 'r', 'y']

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.show()


