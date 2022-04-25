from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

#creating function to create random data
def create_data(amount, min, max):
    nums = []
    count = 0
    while count<amount:
        random_num = np.random.randint(min,max+1)
        nums.append(random_num)
        count +=1
    return nums

#use function to create a dataset and put it into a dataframe
a = create_data(400, 1, 1000)
b = create_data(400, 1, 100)
c = create_data(400, 1, 300)
d = create_data(400, 1, 10)
e = create_data(400, 1, 50)

X ={'a':a, 'b':b, 'c':c, 'd':d, 'e':e}

df = pd.DataFrame(X)

# #check the first rows
# print(df.head())

#normalize the data to scale equaliy
data_scaled = normalize(df)
data_scaled = pd.DataFrame(data_scaled, columns=df.columns)

# #check the normalized data
# print(data_scaled.head())

# #plot the dendrogram to find the best amount of clusters
# plt.figure(figsize=(10,7))
# dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
# plt.show()

#after finding the bes amount of cluster the following function can be used
def dend_best_num(num_cluster, dataframe):
    cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
    cluster.fit_predict(dataframe)
    return cluster

# #show the labels
# print(dend_best_num(2,data_scaled))

#plot the clusters
cluster=dend_best_num(9,data_scaled)
plt.figure(figsize=(10,7))
plt.scatter(data_scaled['c'], data_scaled['d'], c=cluster.labels_)
plt.show()