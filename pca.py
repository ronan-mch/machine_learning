import numpy as np
import pandas
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
from pylab import *

with open('titanic_processed.csv') as f:
   data_frame = pandas.read_csv(f)
   
   # Replace empty values in the age column with the mean of that column
  # for i in range(len(data_frame)):
  #     if math.isnan(data_frame.iloc[i,7]):
  #         data_frame.iloc[i,7] = data_frame.iloc[:,7].mean()

   # Center the values by subtracting the mean and dividing by the standard
   # deviation
   centered = (data_frame
        .fillna(data_frame.mean()) # replace empty values with mean
        .assign(survived_cent = lambda x: ((x.Survived - x.Survived.mean()) /
            x.Survived.std()))
        .assign(died_cent = lambda x: ((x.Died - x.Died.mean()) / x.Died.std()))
        .assign(first_class_cent = lambda x: ((x.FirstClass - x.FirstClass.mean()) / x.FirstClass.std()))
        .assign(second_class_cent = lambda x: ((x.SecondClass - x.SecondClass.mean()) / x.SecondClass.std()))
        .assign(third_class_cent = lambda x: ((x.ThirdClass - x.ThirdClass.mean()) / x.ThirdClass.std()))
        .assign(female_cent = lambda x: ((x.Female - x.Female.mean()) / x.Female.std()))
        .assign(male_cent = lambda x: ((x.Male - x.Male.mean()) / x.Male.std()))
        .assign(age_cent = lambda x: ((x.Age - x.Age.mean()) / x.Age.std()))
        .assign(sibsp_cent = lambda x: ((x.SibSp - x.SibSp.mean()) / x.SibSp.std()))
        .assign(parch_cent = lambda x: ((x.Parch - x.Parch.mean()) / x.Parch.std()))
        .assign(cherbourg_cent = lambda x: ((x.Cherbourg- x.Cherbourg.mean()) / x.Cherbourg.std()))
        .assign(queenstown_cent = lambda x: ((x.Queenstown- x.Queenstown.mean()) / x.Queenstown.std()))
        .assign(southhampton_cent = lambda x: ((x.Southhampton- x.Southhampton.mean()) / x.Southhampton.std()))
        .drop('PassengerId', 1)
        .drop('Survived', 1)
        .drop('Died', 1)
        .drop('FirstClass', 1)
        .drop('SecondClass', 1)
        .drop('ThirdClass', 1)
        .drop('Female', 1)
        .drop('Male', 1)
        .drop('Age', 1)
        .drop('SibSp', 1)
        .drop('Parch', 1)
        .drop('Fare', 1)
        .drop('Cherbourg', 1)
        .drop('Queenstown', 1)
        .drop('Southhampton', 1)
        )
centeredMatrix = np.mat(centered)


# PCA by computing SVD of Y
U,S,V = linalg.svd(centeredMatrix,full_matrices=False)
V = np.mat(V).T

# Project the centered data onto principal component space
K = centeredMatrix * V
#print np.size(K,0)
#print np.size(K,1)

# Compute variance explained by principal components
var = (S*S) / (S*S).sum() 
print sum(var[:2]),"The amount of variation explained as a function of two PCA"
# 
# # Plot PCA of the data
# 
# plt.title('PCA')
# plt.plot(K[:,0], K[:,1], '.', color='red')    #first component
# #plt.plot(K[1,:], K[0,:], '*', color='yellow')   #second component
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# 
# # Output result to screen
# plt.show()
