
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
    # We need to read Fare as a string to enable our string based cleaning method
    data = pandas.read_csv(f, dtype={'Embarked': 'category','Fare': str})
    with_titles = add_titles(data)
    with_fam_size = with_titles.assign(FamilySize = lambda row: row.Parch + row.SibSp)
    with_cleaned_fare = add_cleaned_fare(with_fam_size)
    filled_ages = estimate_ages(with_cleaned_fare)
    return filled_ages

def linear():
    d = engineered()
    pruned = d.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'SibSp', 'Parch'], axis=1)
    binarized = pruned.replace(to_replace={'Sex': {'male': 0, 'female': 1}})
    encoded = pandas.get_dummies(binarized, prefix={'Embarked': 'Emb', 'Title': 'Tit'})
    return encoded

def add_titles(input_data):
    titles = input_data.Name.str.extract('(Mr|Mrs|Miss|Master|Don|Captain|Col|Rev|Ms|Mme|Dr|Major|Countess|Capt|Mlle|Jonkheer)', expand=False)
    titles.name = 'Title'
    return input_data.join(titles)

def estimate_ages(input_data):
    # These are the only titles where there are missing age values
    for title in ['Mr', 'Miss', 'Master', 'Dr']:
        titles_median_age = input_data[input_data.Title == title].Age.median()
        input_data.loc[(input_data.Age.isnull()) & (input_data.Title == title), 'Age'] = titles_median_age
    return input_data

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

