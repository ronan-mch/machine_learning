import titanic_data
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def score_nn(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    nn_regr = MLPRegressor()
    nn_regr.fit(X_train, y_train)
    return nn_regr.score(X_test, y_test)

def score_regr(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    return regr.score(X_test, y_test)

def score_model(train, test, model_features, test_feature, test_func):
    # Attempting to test an empty feature set will cause an error
    if len(model_features) == 0: return 0
    return test_func(train[model_features].values, test[model_features].values, train[test_feature].values, test[test_feature].values)

def test_holdout(data, features, test_func):
    training = data.iloc[:-20]
    test = data.iloc[-20:]
    score = score_model(training, test, features, 'CleanedFare', test_func)
    return score

def test_kfold(data, features, test_func):
    k = 3
    kf = KFold(n_splits=k)
    total_score = 0
    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        score = score_model(train, test, features, 'CleanedFare', test_func)
        total_score += score
    return total_score / k

def compare_features(data, test_func, est_func=test_kfold, baseline=None, cur_model=[], try_features=[]):
    scores = []
    for feature in try_features:
        try_model = cur_model + [feature]
        score = est_func(data, try_model, test_func)
        items = [feature, score]
        scores.append(items)

    best_feature = None
    best_score = baseline
    for f,s in scores:
        if s > best_score:
            best_feature = f
            best_score = s

    return best_feature, best_score


def forward_select(data, current_model, remaining_features, baseline, test_func, est_func=test_kfold, scores=[]):
    if len(remaining_features) == 0:
        print "best model is {0}".format(current_model)
        return current_model, scores
    else:
        bf, bs = compare_features(data, test_func, est_func, baseline=baseline, cur_model=current_model, try_features=remaining_features)
        if bf == None:
            print "New features could not improve on baseline, returning"
            return current_model, scores
        improved_model = current_model + [bf]
        print "best feature is {0}".format(bf)
        features_left = remaining_features.drop(bf)
        scores.append([improved_model, bs])
        return forward_select(data, improved_model, features_left, bs, test_func, scores=scores)

def table_display(labels, scores, gen_err):
    print "============================================"
    for idx, i in enumerate(range(len(labels))):
        print "{0} & {1} & {2} & {3} \\ \hline".format(idx + 1, labels[i], scores[i], gen_err[i])
    print "============================================"

def plot_scores(axis, scores, title, label):
    plot.sca(axis)
    plot.xticks(range(len(scores)), map(lambda x: 1 + x, range(len(scores))))
    axis.plot(scores, label=label)
    axis.legend()
    axis.set_title(title)
    axis.grid(True)

def process_scorechart(scorechart):
    label_arrays,scores = zip(*scorechart)
    labels = [",".join(x) for x in label_arrays]
    gen_errors = map(lambda x: 1 - x, scores)

    return labels, scores, gen_errors

def get_baseline(data, test_func=score_regr):
    y_train = data.CleanedFare.values[:-20]
    y_test = data.CleanedFare.values[-20:]
    return test_func(np.zeros((821, 1)), np.zeros((20, 1)), y_train, y_test)

def estimate_generalisation_errors(scorechart, data, gen_est_data, test_func):
    models, scores = zip(*scorechart)
    gen_est_errors = map(lambda features: 1 - score_model(data, gen_est_data, features, 'CleanedFare', test_func), models)
    return gen_est_errors

