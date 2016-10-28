## Regression problem

For a regression problem I decided to predict Fare based on the other attributes. One reason for
choosing Fare is that it is the only continuous attribute: there are other interval attributes such
as Parch (number of Parents or Children) and Sibsp (number of siblings) but these are discrete,
i.e. they increase in intervals of one.
Obviously we could round the results of a linear regression to return discrete values, but I
preferred not to add this layer of processing. I also assumed that Fare would be easier to predict
based on other attributes such as Pclass and Embarked.

## Value transformation and Feature Engineering

### The Fare attribute

The Fare attribute was a problematic attribute because of a large amount of noisy data. This can be
seen in Figure X (TODO - plot Fare boxplots, before and after) where one can see that a large
number of attributes have unrealistic values. In order to use it for a regression I would thus need to clean the attribute, otherwise, our model would be fitted on incorrect data. Before cleaning the data I needed to determine a couple of things:

    1. What are realistic values for the dataset?
    2. What values can be said to be erroneous?

I decided to compare the values in the dataset to the historical prices for tickets on the Titanic.
I found that the prices in the dataset are transformations of the original prices in British
pounds, shillings and pence. These three values have been transformed to a single value with the shilling and pence converted to decimal points of a pound. One source reports the ticket prices as ranging from £3 for a third class ticket to £870 for a first class ticket. This tallies with the dataset: the median value for the Fare attribute is 19.50, while the 75th percentile is 56.92. Given that 870 was the maximum ticket price, then we can assume that any value over 1000 is erroneous, with the assumption that the Fare price could include other purchases.

If we can assume that the dataset is a noisy representation of the historical data, then how should we treat the outliers? I took a sample of many outlying values and found that the price paid by the actual passengers corresponded to the price represented in the dataset with the exception that the values in the dataset were missing a decimal point! When I discovered this, it was relatively trivial to write a transformation function that would add a decimal point in the correct position. Once applied to the values, I got a more normal distribution as one would expect given the historical context. This can be seen in Fig X (TODO - make a boxplot of the Fare distribution).

### Other attributes

Inspired by other analyses of the same dataset, I decided to engineer a Title attribute based on the Name attribute. The Name attribute is well structured and always contains a title of some sort. These titles can be indicative of a passenger's age, sex, marital status, profession and social status and as such can be assumed to be of relevance in determining their fare. Creating the attribute was a simple matter of identifying all possible values and parsing these out from the text of the name attribute. This code can be seen in the titanic_data.py script.

I also decided to create a mixed FamilySize attribute which would consist of the sum of the SibSp and the Parch attributes to give a single attribute indicating the passenger's familial relations. Furthermore, I binarized the sex value and used one in k encoding on the Title and Embarked categorical values.

## Forward select

Once I had created the engineered features and encoded the categorical features, I was able to perform the linear regression. Using a leave one out validation method I used forward selection to derive the following model. As can be seen in Fig X (TODO) the first few attributes greatly increase the model's accuracy while subsequent attributes contribute less and less to the model. The results can be seen in the following table:

| Attribute | Variance | Improvement on previous model
| ----------------------------------------------------
| None | 0.0176921603903 | 0.0176921603903
| Survived | 0.261357025469 | 0.243664865079
| Survived,FamilySize | 0.396732481369 | 0.1353754559
| Survived,FamilySize,Age | 0.446957468467 | 0.050224987098
| Survived,FamilySize,Age,Emb_Q | 0.457260964935 | 0.0103034964683
| Survived,FamilySize,Age,Emb_Q,Tit_Mr | 0.464184251758 | 0.00692328682227
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex | 0.467945699981 | 0.00376144822344
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess | 0.471576863133 | 0.00363116315198
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle | 0.474518315039 | 0.00294145190604
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme | 0.474744309095 | 0.000225994056255
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme,Tit_Major | 0.475366629929 | 0.000622320833382
| Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme,Tit_Major,Tit_Rev | 0.47537164988 | 5.0199507573e-06

There are some interesting results from the exploration process: the Pclass attribute is not included at all in the final model. This seems paradoxical since that attribute is the one that would be assumed to have most relevance for the Fare attribute. Looking at the engineered features only, it can be seen that the FamilySize feature improves the model's accuracy significantly. I decided to run the same test without any engineered features to see how the results compare. As shown in the following table, a similar accuracy is obtained using only two features, Survived and Age. This is quite interesting because it illustrates a quirk of forward selection, by using a non-exhaustive testing method we were returned a model with a similar accuracy but much higher dimensionality while an almost equally good model was not tested at all.
 
 None | 0.0176921603903 | 0.0176921603903
 Survived | 0.215949183526 | 0.198257023136
 Survived,Age | 0.462524575678 | 0.246575392152

## Forward selection - KFold vs Holdout

The choice of method for estimating the generalisation error has significant consequences for the
result of model selection. Comparing the outcome of a model selection using K-Fold vs Holdout we
can see that the K-Fold method returned a smaller model with a lower confidence than that returned
by the Holdout method. For our next trick we will compare the performance of the models using two
layer cross validation.

The above results were achieved with a simple holdout method of estimating the generalization
error. Here we present the results when using the KFold method.

### K-Fold

None | 0.695943488954 | 0.695943488954
Cls_first | 0.616624557562 | -0.0793189313919
Cls_first,FamilySize | 0.610492356137 | -0.00613220142498
Cls_first,FamilySize,Cls_second | 0.606225285838 | -0.00426707029911
Cls_first,FamilySize,Cls_second,Emb_S | 0.603275922738 | -0.00294936310002
Cls_first,FamilySize,Cls_second,Emb_S,Tit_Major | 0.602784933223 | -0.000490989515368
Cls_first,FamilySize,Cls_second,Emb_S,Tit_Major,Tit_Master | 0.600742676739 | -0.00204225648426
Cls_first,FamilySize,Cls_second,Emb_S,Tit_Major,Tit_Master,Age | 0.600621848603 | -0.00012082813537
Cls_first,FamilySize,Cls_second,Emb_S,Tit_Major,Tit_Master,Age,Tit_Countess | 0.600579356421 |
-4.24921825224e-05

### Holdout

None | 0.98230783961 | 0.98230783961
Survived | 0.738642974531 | -0.243664865079
Survived,FamilySize | 0.603267518631 | -0.1353754559
Survived,FamilySize,Age | 0.553042531533 | -0.050224987098
Survived,FamilySize,Age,Emb_Q | 0.542739035065 | -0.0103034964683
Survived,FamilySize,Age,Emb_Q,Tit_Mr | 0.535815748242 | -0.00692328682227
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex | 0.532054300019 | -0.00376144822344
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess | 0.528423136867 | -0.00363116315198
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle | 0.525481684961 | -0.00294145190604
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme | 0.525255690905 |
-0.000225994056255
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme,Tit_Major | 0.524633370071 |
-0.000622320833382
Survived,FamilySize,Age,Emb_Q,Tit_Mr,Sex,Tit_Countess,Tit_Mlle,Tit_Mme,Tit_Major,Tit_Rev |
0.52462835012 | -5.0199507573e-06

