import titanic_data as data
import matplotlib.pyplot as plot
import pandas as pd

#data.engineered().CleanedFare.plot.kde(title="Fare density")
# plot.savefig("fare_density.png")
# data.original.Sex.value_counts().plot.bar(title="Sex distribution")
# plot.savefig("sex_dist.png")
# data.original.Embarked.value_counts().plot.bar(title="Embarked distribution")
# plot.savefig("embarked_dist.png")
# data.original.Age.plot.kde(title = "Age distribution")
# plot.savefig("age_dist.png")
#data.engineered().Title.value_counts().plot.bar(title="titles")
d = data.engineered()
male_famsize = d[d['Sex'] == "male"].FamilySize
female_famsize = d[d['Sex'] == "female"].FamilySize

queenstown_famsize = d[d.Embarked == 'Q'].FamilySize
cherbourg_famsize = d[d.Embarked == 'C'].FamilySize
southhampton_famsize = d[d.Embarked == 'S'].FamilySize

first_class_famsize = d[d.Pclass == 1].FamilySize
second_class_famsize = d[d.Pclass == 2].FamilySize
third_class_famsize = d[d.Pclass == 3].FamilySize

sex_con = pd.concat([male_famsize, female_famsize], join_axes=None, axis=1, keys=["Male", "Female"])
embarked_con = pd.concat([queenstown_famsize, cherbourg_famsize, southhampton_famsize],
        join_axes=None, axis=1, keys=["Queenstown", "Cherbourg", "Southhampton"])
class_con = pd.concat([first_class_famsize, second_class_famsize, third_class_famsize],
        join_axes=None, axis=1, keys=["First", "Second", "Third"])
#class_con.plot.box()

queenstown_fare = d[d.Embarked == 'Q'].CleanedFare
cherbourg_fare = d[d.Embarked == 'C'].CleanedFare
southhampton_fare = d[d.Embarked == 'S'].CleanedFare

first_class_fare = d[d.Pclass == 1].CleanedFare
second_class_fare = d[d.Pclass == 2].CleanedFare
third_class_fare = d[d.Pclass == 3].CleanedFare


f, (ax_one, ax_two) = plot.subplots(1, 2)
uncleaned = d.Fare.astype('float64').values
cleaned = d.CleanedFare.values
ax_one.boxplot(uncleaned, labels=["Fare"])
ax_one.set_title("Original Values")
ax_two.boxplot(cleaned, labels=["CleanedFare"])
ax_two.set_title("Cleaned Values")
# pd.DataFrame(fare_data).plot.box(showmeans=True, showbox=True) 
plot.show()
#data.engineered().plot.bar(y='Sex', x='Parch')

