
import pandas

with open('titanic_train.csv') as f:
    original = pandas.read_csv(f)

with open('titanic_processed.csv') as f:
   processed = pandas.read_csv(f)
   processed = processed.drop(processed.columns[0], 1)

