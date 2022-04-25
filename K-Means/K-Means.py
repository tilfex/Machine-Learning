import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#create lists for the dataset
x = []
y = []

#create a random x-y-dataset
for i in range(0,20):
    num_x = np.random.randint(1,101)
    num_y = np.random.randint(1,101)
    x.append(num_x)
    y.append(num_y)

#create a dictionary for the created dataset
Data = {'x':x,'y':y}

#create a dataframe with the datasets
df = pd.DataFrame(Data, columns=['x','y'])

#creation and fitting of the K-Means-Model
kmeans = KMeans(n_clusters=3).fit(df)

#finding the centers
centers = kmeans.cluster_centers_
# print (centers)

#getting cluster_labels for each datapair
pair_labels = kmeans.labels_
# print(pair_labels)

#plotting the clusters and its centers
plt.scatter(df['x'],df['y'], c = pair_labels.astype(float), s=50, alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)
plt.show()