import titanic_data 
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plot

def score_model(X_train, X_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    return regr.score(X_test, y_test)

def test_features(features): 
    X_training = data[features].values[:-20]
    X_test = data[features].values[-20:]
    score = score_model(X_training, X_test)
    return score

def compare_features(baseline=None, cur_model=[], try_features=[]):
    scores = []
    for feature in try_features:
        try_model = cur_model + [feature]
        score = test_features(try_model)
        items = [feature, score]
        scores.append(items)
    
    best_feature = None
    best_score = baseline
    for f,s in scores:
        if s > best_score:
            # print "{0} is better than {1}!!!".format(cur_model + [f], best_feature) 
            best_feature = f
            best_score = s

    return best_feature, best_score


def forward_select(current_model, remaining_features, baseline, scores=[]):
    if len(remaining_features) == 0:
        print "best model is {0}".format(current_model)
        return current_model, scores
    else:
        bf, bs = compare_features(baseline=baseline, cur_model=current_model,
                try_features=remaining_features)
        print "{0}: {1}".format(current_model, bs)
        print "scores are {0}".format(scores)
        if bf == None: 
            print "New features could not improve on baseline, returning"
            return current_model, scores
        improved_model = current_model + [bf]
        features_left = remaining_features.drop(bf)
        scores.append([current_model, bs])
        return forward_select(improved_model, features_left, bs, scores=scores)

data = titanic_data.linear()
y_train = data.CleanedFare.values[:-20]
y_test = data.CleanedFare.values[-20:]
root_baseline = score_model(np.zeros((871, 1)), np.zeros((20, 1)))
predictor_features = data.columns.drop('CleanedFare')
model = []
selected_model, scorechart = forward_select(model, predictor_features, root_baseline, [])
label_arrays,scores = zip(*scorechart) 
label_arrays[0].append("None")
labels = [",".join(x) for x in label_arrays]
print "labels are {0} and scores are {1}".format(labels, scores)
plot.plot(scores)
plot.xticks(range(len(labels)), labels, rotation=90)
plot.grid(True)
plot.yticks()
plot.show()


