# Machine Learning

This repository consists of data and scripts for our machine learning assignment(s). The dataset we have chosen is the [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle. This dataset contains a list of passengers on the Titanic together with several of their characteristics including age, sex, seating class, the price paid for their ticket and whether or not they survived.

## Dataset

The original dataset consists of the following attributes:

| Attribute    | Cardinality | Type    | Notes |
|--------------|-------------|---------|-------|
| Passenger id | Discrete    | Nominal |       |
| Survived     | Binary      | Nominal |       |
| Pclass       | Discrete    | Ordinal |       |
| Name         | ?           | Nominal |       |
| Sex          | Binary      | Nominal |       |
| Age          | Discrete    | Ratio   | 177 missing vals |
| Sibsp        | Discrete    | Ratio   | Refers to number of siblings onboard |
| Parch        | Discrete    | Ratio   | Refers to number of parents or children aboard |
| Ticket       | Discrete    | Nominal |       |
| Fare         | Continuous  | Ratio   |       |
| Cabin        | Discrete    | Nominal |       |
| Embarked     | Discrete    | Nominal | Can be Q (Queenstown, Ireland), C (Cherbourg, France) or S (Southampton, England). 2 missing vals|

## Processing

 1. The attributes name and ticket were deemed unnecessary to the analysis and were removed. The attribute cabin was removed since it had very few datapoints (only 204 out of 891 values).
 2. The Sex and Embarked attributes were encoded using 1-in-K encoding, thus adding two additional columns, Male and Female and three additional columns, Queenstown, Cherbourg and Southhampton respectively. The original columns were then removed. 
 3. A simple python script was used to normalise the data by first subtracting the mean of the given column from each value, then dividing the resulting value by the variance of that column.
