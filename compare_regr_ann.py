import titanic_data
import forward_select as fwd
import matplotlib.pyplot as plot
import numpy as np
import sklearn.metrics as metrics

def main():
    linear = titanic_data.linear()
    data = linear[:-89]
    gen_est_data = linear[-89:]
    regr_baseline = fwd.get_baseline(data)
    predictor_features = data.columns.drop('CleanedFare')

    # Linear Regression
    lin_best, lin_scorechart = fwd.forward_select(data, [], predictor_features, regr_baseline, fwd.score_regr, scores=[])
    lin_labels, lin_scores, lin_error_rates = fwd.process_scorechart(lin_scorechart)
    lin_gen_est_errors = fwd.estimate_generalisation_errors(lin_scorechart, data, gen_est_data, fwd.score_regr)
    fwd.table_display(lin_labels, lin_scores, lin_gen_est_errors)

    # Assume average Fare
    avg_fare = np.mean(data.CleanedFare.values)
    avg_arr = np.empty(len(gen_est_data))
    avg_arr.fill(avg_fare)
    avg_arr.reshape(1,-1)
    print("avg_arr", avg_arr)
    avg_fare_error =  1 - metrics.r2_score(gen_est_data.CleanedFare.values, avg_arr)
    print("Average:", avg_fare)
    print("Average Error:", avg_fare_error)

    # Neural Network
    nn_baseline = fwd.get_baseline(data, test_func=fwd.score_nn)
    nn_best, nn_scorechart = fwd.forward_select(data, [], predictor_features, nn_baseline, test_func=fwd.score_nn, est_func=fwd.test_kfold, scores=[])
    nn_labels, nn_scores, nn_error_rates = fwd.process_scorechart(nn_scorechart)
    nn_gen_est_errors = fwd.estimate_generalisation_errors(nn_scorechart, data, gen_est_data, fwd.score_nn)
    fwd.table_display(nn_labels, nn_scores, nn_gen_est_errors)


    # Graphing code
    f, axarr = plot.subplots(1, 2, sharey=True)

    fwd.plot_scores(axarr[0], lin_scores, "Linear Regression", "Train")
    fwd.plot_scores(axarr[0], lin_gen_est_errors, "Linear Regression", "Test")
    avg_fare_err_arr = np.empty(len(lin_scores))
    avg_fare_err_arr.fill(avg_fare_error)
    axarr[0].plot(avg_fare_err_arr, label="Assume Average Fare")
    axarr[0].legend()

    fwd.plot_scores(axarr[1], nn_scores, "Neural Network", "Train")
    fwd.plot_scores(axarr[1], nn_gen_est_errors, "Neural Network", "Test")
    avg_fare_err_arr = np.empty(len(nn_scores))
    avg_fare_err_arr.fill(avg_fare_error)
    axarr[1].plot(avg_fare_err_arr, label="Assume Average Fare")
    axarr[1].legend()

    f.text(0.5, 0.04, 'Model index', ha='center', va='center')
    f.text(0.06, 0.5, 'Generalisation error', ha='center', va='center', rotation='vertical')

    plot.show()

if __name__ == "__main__":
    main()
