
import pandas

def original():
    with open('titanic_train.csv') as f:
        return pandas.read_csv(f)

def processed():
    with open('titanic_processed.csv') as f:
       processed = pandas.read_csv(f)
       return processed.drop(processed.columns[0], 1)

def engineered():
    data = original()
    titles = data.Name.str.extract('(Mr|Mrs|Miss|Master|Don|Captain|Col|Rev|Ms|Mme|Dr|Major|Countess|Capt|Mlle|Jonkheer)', expand=False)
    titles.name = 'Title'
    return data.join(titles)

