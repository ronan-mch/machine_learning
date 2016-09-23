import titanic_data as data
import matplotlib.pyplot as plot

data.original.Fare.plot.kde(title="Fare density")
plot.savefig("fare_density.png")
data.original.Sex.value_counts().plot.bar(title="Sex distribution")
plot.savefig("sex_dist.png")
data.original.Embarked.value_counts().plot.bar(title="Embarked distribution")
plot.savefig("embarked_dist.png")
data.original.Age.plot.kde(title = "Age distribution")
plot.savefig("age_dist.png")

