import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

# creating dataset
x = np.random.normal(size=150)
y = (x > 0).astype(np.float)
x[x > 0] *= 4
x += 0.3 * np.random.normal(size=150)
x = x.reshape(-1, 1)
x_ordered = np.sort(x, axis=0)

# #plot the data
# plt.scatter(x, y)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()

# logistic regression analzsis
l_regr = linear_model.LogisticRegression()
l_regr.fit(x, y)

# plot the model
plt.scatter(x.ravel(), y, color='black', zorder=20, alpha=0.5)
plt.plot(x_ordered, l_regr.predict_proba(x_ordered)[:, 1], color='blue', linewidth=3)
plt.show()