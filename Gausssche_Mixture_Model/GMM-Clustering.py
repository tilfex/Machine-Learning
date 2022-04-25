import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture

#loading iris dataset
iris_data = datasets.load_iris()

#choose the first two columns
x = iris_data.data[:, :2]

#push data into a dataframe
df = pd.DataFrame(x)

# #plot the data
# plt.scatter(df[0], df[1])
# plt.show()

#partitioning and fitting the data
gmm = GaussianMixture(n_components= 3)
gmm.fit(df)

#label the datapairs
labels = gmm.predict(df)

# #check convergence of model
# print('Converged: ' ,gmm.converged_)

# #check the amount of needed iterations
# print(gmm.n_iter_)

#getting the means
means = gmm.means_
# print(means)

#getting covariances
covariances = gmm.covariances_
# print(covariances)

#plot the clusters
df['labels']= labels
df0 = df[df['labels']== 0]
df1 = df[df['labels']== 1]
df2 = df[df['labels']== 2]
plt.scatter(df0[0], df0[1], c='red')
plt.scatter(df1[0], df1[1], c='yellow')
plt.scatter(df2[0], df2[1], c='green')
plt.show()
