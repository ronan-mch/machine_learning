# Machine Learning

This repository consists of data and scripts for our machine learning assignment(s). The dataset we have chosen is the [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle. This dataset contains a list of passengers on the Titanic together with several of their characteristics including age, sex, seating class, the price paid for their ticket and whether or not they survived.

## Dataset

The original dataset consists of the following attributes:

| Attribute    | Cardinality | Type    |
|--------------|-------------|---------|
| Passenger id | Discrete    | Nominal |
| Survived     | Binary      | Nominal |
| Pclass       | Discrete    | Ordinal |
| Name         | ?           | Nominal |
| Sex          | Binary      | Nominal |
| Age          | Discrete    | Ratio   |
| Sibsp        | Discrete    | Ratio   |
| Parch        | Discrete    | Ratio   |
| Ticket       | Discrete    | Nominal |
| Fare         | Continuous  | Ratio   |
| Cabin        | Discrete    | Nominal |
| Embarked     | Discrete    | Nominal |

## Processing

 1. The attributes name and ticket were deemed unnecessary to the analysis and were removed. The attribute cabin was removed since it had very few datapoints (only 204 out of 891 values).
 2. The sex attribute was converted from a text string to a binary attribute, where 1 represents male and 0 represents female.
