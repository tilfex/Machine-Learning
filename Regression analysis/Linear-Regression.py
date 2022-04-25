import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

#loading the dataset
bostonData = datasets.load_boston()
y = bostonData.target.reshape(-1,1)
x = bostonData['data'][:, 5].reshape(-1, 1)

# #plot the variables
# plt.scatter(x, y)
# plt.ylabel('Value of house / 1000 [$]')
# plt.xlabel('Number of rooms')
# plt.show()

#create the model
regr = linear_model.LinearRegression()

#train the model
regr.fit(x,y)

#generate the plots
plt.scatter(x, y, color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
plt.show()