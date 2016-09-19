import numpy
import pandas
import matplotlib.pyplot as plt

with open('titanic_processed.csv') as f:
   data_frame = pandas.read_csv(f)
   centered = (data_frame
        .assign(survived_cent = lambda x: ((x.Survived - x.Survived.mean()) / x.Survived.var()))
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
   print centered
      

