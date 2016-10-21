import titanic_data as data
import matplotlib.pyplot as plot
import pandas as pd
from sklearn import linear_model
import numpy as np

d = data.linear()
titanic_X_train = d[['Pclass', 'Embarked_C']].values[:-20]
titanic_X_test = d[['Pclass', 'Embarked_C']].values[-20:]

titanic_y_train = d.CleanedFare.values[:-20]
titanic_y_test = d.CleanedFare.values[-20:]

regr = linear_model.LinearRegression()
regr.fit(titanic_X_train, titanic_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(titanic_X_test) - titanic_y_test) ** 2))

# Explained variance score
print("Variance score %.2f" % regr.score(titanic_X_test, titanic_y_test))
print regr.predict(titanic_X_test)
# plot.scatter(titanic_X_test, titanic_y_test, color="black")
# plot.plot(titanic_X_test, regr.predict(titanic_X_test), color="blue", linewidth=3)
# plot.xticks()
# plot.yticks()
# plot.show()
