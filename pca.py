import numpy as np
import pandas
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
from pylab import *

with open('titanic_processed.csv') as f:
   data_frame = pandas.read_csv(f)
   print data_frame
   data_frame = data_frame.drop(data_frame.columns[0], 1)
   print data_frame
   
   for i in range(len(data_frame)):
       if math.isnan(data_frame .iloc[i,4]):
           data_frame .iloc[i,4] = data_frame .iloc[:,4].mean()
   centered = (data_frame
        .assign(survived_cent = lambda x: ((x.Survived - x.Survived.mean()) / x.Survived.std())
        .assign(class_cent = lambda x: ((x.Pclass - x.Pclass.mean()) / x.Pclass.var()))
        .assign(female_cent = lambda x: ((x.Female - x.Female.mean()) / x.Female.var()))
        .assign(male_cent = lambda x: ((x.Male - x.Male.mean()) / x.Male.var()))
        .assign(age_cent = lambda x: ((x.Age - x.Age.mean()) / x.Age.var()))
        .assign(sibsp_cent = lambda x: ((x.SibSp - x.SibSp.mean()) / x.SibSp.var()))
        .assign(parch_cent = lambda x: ((x.Parch - x.Parch.mean()) / x.Parch.var()))
        .assign(fare_cent = lambda x: ((x.Fare- x.Fare.mean()) / x.Fare.var()))
        .assign(cherbourg_cent = lambda x: ((x.Cherbourg- x.Cherbourg.mean()) / x.Cherbourg.var()))
        .assign(queenstown_cent = lambda x: ((x.Queenstown- x.Queenstown.mean()) / x.Queenstown.var()))
        .assign(southhampton_cent = lambda x: ((x.Southhampton- x.Southhampton.mean()) / x.Southhampton.var()))
        )
   #print centered
   #print centered.iloc[:,5]
   
   
# PCA by computing SVD of Y
#U,S,V = linalg.svd(centered,full_matrices=False)
#V = np.mat(V).T
#print V

# Project the centered data onto principal component space
#Z = centered * V
#print Z

'''

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
f.hold()
title('Titanic dataset: PCA')
Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
'''