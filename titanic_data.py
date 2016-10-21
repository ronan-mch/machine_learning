
import pandas

def original():
    with open('titanic_train.csv') as f:
        return pandas.read_csv(f)

def processed():
    with open('titanic_processed.csv') as f:
       processed = pandas.read_csv(f)
       return processed.drop(processed.columns[0], 1)

def engineered():
    f = open('titanic_train.csv')
    data = pandas.read_csv(f, dtype={'Embarked': 'category','Fare': str})
    with_titles = add_titles(data)
    with_fam_size = with_titles.assign(FamilySize = lambda row: row.Parch + row.SibSp)
    with_cleaned_fare = add_cleaned_fare(with_fam_size)
    return with_cleaned_fare

def linear():
    d = engineered() 
    pruned = d.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
    encoded = pandas.get_dummies(pruned)
    return encoded

def add_titles(input_data): 
    titles = input_data.Name.str.extract('(Mr|Mrs|Miss|Master|Don|Captain|Col|Rev|Ms|Mme|Dr|Major|Countess|Capt|Mlle|Jonkheer)', expand=False)
    titles.name = 'Title'
    return input_data.join(titles)

def add_cleaned_fare(input_data):
    cleaned_fare = input_data.Fare.apply(add_dot).astype('float64')
    cleaned_fare.name = 'CleanedFare'
    return input_data.join(cleaned_fare)

# insert a dot into strings without decimals 
def add_dot(string):
    if "." in string: return string
    if string == "0": return string
    if int(string) <= 999: return string
    return string[:-3] + "." + string[-3:]
