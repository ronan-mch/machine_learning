import titanic_data
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plot

def score_regr(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    return regr.score(X_test, y_test)

def score_model(train, test, model_features, test_feature):
    return score_regr(train[model_features].values, test[model_features].values, train[test_feature].values, test[test_feature].values)

def test_holdout(features):
    training = data.iloc[:-20]
    test = data.iloc[-20:]
    score = score_model(training, test, features, 'CleanedFare')
    return score

def test_kfold(features):
    kf = KFold(n_splits=10)
    total_score = 0
    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        score = score_model(train, test, features, 'CleanedFare')
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
        return current_model, scores
    else:
        bf, bs = compare_features(test_func, baseline=baseline, cur_model=current_model,
                try_features=remaining_features)
        if bf == None: 
            print "New features could not improve on baseline, returning"
            return current_model, scores
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

def plot_scores(axis, scores, title, label):
    plot.sca(axis)
    plot.xticks(range(len(scores)), map(lambda x: 1 + x, range(len(scores))))
    axis.plot(scores, label=label)
    axis.legend()
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
    return score_regr(np.zeros((821, 1)), np.zeros((20, 1)), y_train, y_test)

def estimate_generalisation_errors(scorechart):
    models, scores = zip(*scorechart)
    gen_est_errors = map(lambda x: 1 - score_model(data, gen_est_data, x, 'CleanedFare'), models[1:])
    gen_est_errors.insert(0, 1)
    return gen_est_errors

linear = titanic_data.linear()
data = linear[:-50]
gen_est_data = linear[-50:]
root_baseline = get_baseline(data)
predictor_features = data.columns.drop('CleanedFare')

# Holdout
model = []
ho_best, ho_scorechart = forward_select(model, predictor_features, root_baseline, test_holdout, [])
ho_labels, ho_scores, ho_error_rates = process_scorechart(ho_scorechart)
table_display(ho_labels, ho_error_rates)
ho_gen_est_errors = estimate_generalisation_errors(ho_scorechart)
print("ho_gen_est is", ho_gen_est_errors)

# K-Fold
model = []
k_best, k_scorechart = forward_select(model, predictor_features, root_baseline, test_kfold, [])
k_labels, k_scores, k_error_rates = process_scorechart(k_scorechart)
table_display(k_labels, k_error_rates)
k_gen_est_errors = estimate_generalisation_errors(k_scorechart)

f, axarr = plot.subplots(1, 2, sharey=True)
plot_scores(axarr[0], ho_error_rates[1:], "Holdout method", "Train")
plot_scores(axarr[0], ho_gen_est_errors[1:], "Holdout method", "Test")

plot_scores(axarr[1], k_error_rates[1:], "K-Fold method", "Train")
plot_scores(axarr[1], k_gen_est_errors[1:], "K-Fold method", "Test")

f.text(0.5, 0.04, 'Model index', ha='center', va='center')
f.text(0.06, 0.5, 'Generalisation error', ha='center', va='center', rotation='vertical')


plot.show()
