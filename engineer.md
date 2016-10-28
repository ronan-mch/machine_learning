## Assumptions of Linear Regression

We need to convert our categorical variables to interval variables. e.g. male / female -> 0 / 1,
C,Q,S -> one in k. Luckily, Pandas has a few helper methods to speed this up: first, we drop the
columns we do not want to use (the dirty Fare variables, the PassengerId, Name, Ticket and Cabim),
then we encode. We are left with 26 columns, 14 of which consist of one-in-k encodings of the
engineered title variable. Let's try these out and see how we get on.

array(['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'FamilySize',
       'CleanedFare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'Title_Capt', 'Title_Col', 'Title_Countess',
       'Title_Don', 'Title_Dr', 'Title_Jonkheer', 'Title_Major',
       'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme', 'Title_Mr',
       'Title_Ms', 'Title_Rev'], dtype=object)
## Feature Engineering

I decided that it was necessary to use the Fare attribute as the target for the linear regression.
This is because the Fare attribute is the only variable that is continuous. All other ratio
attributes are discrete, i.e. a passenger cannot have 1.5 parents or children. As mentioned in the
previous assignment, the Fare attribute is extremely messy with many outliers and we decided to remove it for the
purposes of the PCA. In order to use it for a regression I would thus need to clean the attribute,
otherwise, our model would be fitted on incorrect data. Before cleaning the data I needed to
determine a couple of things:

    1. What are realistic values for the dataset?
    2. What values can be said to be erroneous?

I began by exploring the Fare attribute. As can be seen in the Fare density graph, the vast
majority of observations are clustered around a few values. The median of the set is 19.50, while
the 75th percentile is 56.92. These values correspond closely with the historical data:
Reference.com reports the prices as ranging from £3 for a third class ticket to £870 for a first
class ticket. If we just look at the observations where Fare is greater than 5000 we find 109 observations, where the median is 9,225 and the mean is 32,681. Let's look at the lower end (between 5000 and 29,125 the 75th percentile) to get a sense of the data here. The first example we find is a passenger with Fare 7,925 and PClass 3. This seems unlikely, as the median Fare for third class passage is 9. When we look up the passenger online we can see that she paid £7 (approx $35). Another example from third class is a passenger who apparently paid 21,075 for their fare. Digging a little bit deeper we can see that there are 90 observations with a Pclass of 3 and a Fare over 100. Based on our historical knowledge, we can conclude that these datapoints are more than likely erroneous. In order to correct these we should find some valid replacement values. Grouping our third class passengers by sex shows that there is a large difference between the median prices paid by male and female third class passengers (8.05 vs 15.50), this relationship holds for both first and second class passengers. Thus, the sex of the third class 

Passengers with Fares of 0. There are a fifteen passengers with a fare of 0, all male. These values
seem to be correct, as the passengers concerned served as crew or were official guests of some
type.

Free passengers: 
https://www.encyclopedia-titanica.org/titanic-victim/william-cahoone-johnson.html,
https://www.geni.com/people/Anthony-Wood-Frost/6000000015927969696
https://www.encyclopedia-titanica.org/titanic-victim/johan-george-reuchlin.html

We can expect there to be some standard values corresponding to the different types of tickets and
this appears to be the case. The most common values all correspond to first and second class
tickets. There are also a signigicant number of outliers which differ from the most common values
only in the placement of a decimal point; this leads me to suspect a transcription error. Selecting
all passengers with a Fare of 7925.0 we can see that they all have Pclass 3. Looking up the
passengers shows that they all paid the standard price of £7 18s 6d. Thus we can safely correct
these to use the correct values. The same is true of those passengers with fares of 7775 and 7225.
Looking at the most common values

### Age

There are approx 177 observations that are missing the value for Age. Since the engineered attribute Title has some relevance to Age, I decided to use the mean for that observation's title to fill in the Age variable. Thhis was preferable to dropping the attribute as I presumed it had some significance for both the Fare and the Survived category.
replacing with Age 31.0 for Mr
replacing with Age 21.0 for Miss
replacing with Age 3.5 for Master
replacing with Age 39.0 for Dr

8.0500        43
13.0000       42
7.8958        38
7.7500        34
26.0000       31
10.5000       24
7925.0000     18
7775.0000     16
26.5500       15
7.2292        15
0.0000        15
7.2500        13
7.8542        13
8.6625        13
7225.0000     12
9.5000         9
16.1000        9
24.1500        8
15.5000        8
69.5500        7
31275.0000     7
52.0000        7
7.0500         7
14.5000        7
56.4958        7
14.4542        7
39.6875        6
27.9000        6
46.9000        6
21.0000        6
..
9.8417         1
61.9792        1
81.8583        1
40125.0000     1
8.3000         1
10.5167        1
13.7917        1
221.7792       1
12525.0000     1
28.7125        1
10.1708        1
9475.0000      1
39.4000        1
13.4167        1
76.2917        1
35.0000        1
25925.0000     1
7.0458         1
7.6292         1
26.3875        1
8.8500         1
9.4833         1
51.4792        1
7.7292         1
8.6833         1
8.6542         1
8.4333         1
7.8875         1
32.3208        1
22525.0000     1
 
