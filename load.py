import numpy
import pandas
import matplotlib.pyplot as plt

with open('titanic_processed.csv') as f:
   data_frame = pandas.read_csv(f)
   centered = (data_frame
        .assign(sex_cent = lambda x: (x.Sex - x.Sex.mean()))
        .assign(age_cent = lambda x: (x.Age - x.Age.mean()))
        .assign(sibsp_cent = lambda x: (x.SibSp - x.SibSp.mean()))
        .assign(parch_cent = lambda x: (x.Parch - x.Parch.mean()))
        .assign(fare_cent = lambda x: (x.Fare- x.Fare.mean()))
        )
   print centered

