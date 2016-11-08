import titanic_data
import forward_select as fwd
from sklearn.preprocessing import StandardScaler

linear = titanic_data.linear()
data = linear[:-50]
gen_est_data = linear[-50:]
predictor_features = data.columns.drop('CleanedFare')
nn_baseline = fwd.get_baseline(data, test_func=fwd.score_nn)
nn_best, nn_scorechart = fwd.forward_select(data, [], predictor_features, nn_baseline, test_func=fwd.score_nn, est_func=fwd.test_kfold)
nn_labels, nn_scores, nn_error_rates = fwd.process_scorechart(nn_scorechart)
fwd.table_display(nn_labels, nn_error_rates)
