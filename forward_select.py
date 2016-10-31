import titanic_data
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plot

def score_model(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    return regr.score(X_test, y_test)


def test_holdout(features):
    X_training = data[features].values[:-20]
    X_test = data[features].values[-20:]
    y_train = data.CleanedFare.values[:-20]
    y_test = data.CleanedFare.values[-20:]
    score = score_model(X_training, X_test, y_train, y_test)
    return score

def test_kfold(features):
    kf = KFold(n_splits=10)
    total_score = 0
    for train_index, test_index in kf.split(data):
        X_train, X_test = data.loc[train_index][features], data.loc[test_index][features]
        y_train, y_test = data.loc[train_index]['CleanedFare'], data.loc[test_index]['CleanedFare']
        score = score_model(X_train, X_test, y_train, y_test)
        total_score += score
    return total_score / 10

def compare_features(test_func, baseline=None, cur_model=[], try_features=[]):
    scores = []
    for feature in try_features:
        try_model = cur_model + [feature]
        score = test_func(try_model)
        items = [feature, score]
        scores.append(items)

    best_feature = None
    best_score = baseline
    for f,s in scores:
        if s > best_score:
            best_feature = f
            best_score = s

    return best_feature, best_score


def forward_select(current_model, remaining_features, baseline, test_func, scores=[]):
    if len(remaining_features) == 0:
        print "best model is {0}".format(current_model)
        return scores
    else:
        bf, bs = compare_features(test_func, baseline=baseline, cur_model=current_model,
                try_features=remaining_features)
        if bf == None: 
            print "New features could not improve on baseline, returning"
            return scores
        improved_model = current_model + [bf]
        features_left = remaining_features.drop(bf)
        scores.append([current_model, bs])
        return forward_select(improved_model, features_left, bs, test_func, scores=scores)

def table_display(labels, scores):
    last_val = 0
    for i in range(len(labels)):
        difference = scores[i] - last_val
        print "{0} | {1} | {2}".format(labels[i], scores[i], difference)
        last_val = scores[i]

def plot_scores(axis, scores, title):
    axis.plot(scores)
    axis.set_title(title)
    axis.grid(True)

def process_scorechart(scorechart):
    label_arrays,scores = zip(*scorechart) 
    label_arrays[0].append("None")
    labels = [",".join(x) for x in label_arrays]
    gen_errors = map(lambda x: 1 - x, scores)

    return labels, scores, gen_errors

def get_baseline(data):
    y_train = data.CleanedFare.values[:-20]
    y_test = data.CleanedFare.values[-20:]
    return score_model(np.zeros((871, 1)), np.zeros((20, 1)), y_train, y_test)

data = titanic_data.linear()
root_baseline = get_baseline(data)
predictor_features = data.columns.drop('CleanedFare')

# Holdout
model = []
ho_scorechart = forward_select(model, predictor_features, root_baseline, test_holdout, [])
ho_labels, ho_scores, ho_error_rates = process_scorechart(ho_scorechart)
table_display(ho_labels, ho_error_rates)

# K-Fold
model = []
k_scorechart = forward_select(model, predictor_features, root_baseline, test_kfold, [])
k_labels, k_scores, k_error_rates = process_scorechart(k_scorechart)
table_display(k_labels, k_error_rates)

f, axarr = plot.subplots(1, 2, sharey=True)
plot_scores(axarr[0], ho_error_rates, "Holdout method")
plot_scores(axarr[1], k_error_rates, "K-Fold method")

f.text(0.5, 0.04, 'Model index', ha='center', va='center')
f.text(0.06, 0.5, 'Generalisation error', ha='center', va='center', rotation='vertical')


plot.show()
