import titanic_data
import matplotlib.pyplot as plot
import forward_select as fwd

def main():
    linear = titanic_data.linear()
    data = linear[:-50]
    gen_est_data = linear[-50:]
    root_baseline = fwd.get_baseline(data)
    predictor_features = data.columns.drop('CleanedFare')

    # Holdout
    model = []
    ho_best, ho_scorechart = fwd.forward_select(data, model, predictor_features, root_baseline, fwd.score_regr, fwd.test_holdout, [])
    ho_labels, ho_scores, ho_error_rates = fwd.process_scorechart(ho_scorechart)
    ho_gen_est_errors = fwd.estimate_generalisation_errors(ho_scorechart, data, gen_est_data, fwd.score_regr)
    fwd.table_display(ho_labels, ho_error_rates, ho_gen_est_errors)
    print("ho_gen_est is", ho_gen_est_errors)

    # K-Fold
    model = []
    k_best, k_scorechart = fwd.forward_select(data, model, predictor_features, root_baseline, fwd.score_regr, fwd.test_kfold, [])
    k_labels, k_scores, k_error_rates = fwd.process_scorechart(k_scorechart)
    k_gen_est_errors = fwd.estimate_generalisation_errors(k_scorechart, data, gen_est_data, fwd.score_regr)
    fwd.table_display(k_labels, k_error_rates, k_gen_est_errors)

    f, axarr = plot.subplots(1, 2)
    fwd.plot_scores(axarr[0], ho_scores, "Holdout method", "Train")
    fwd.plot_scores(axarr[0], ho_gen_est_errors, "Holdout method", "Test")

    fwd.plot_scores(axarr[1], k_scores, "K-Fold method", "Train")
    fwd.plot_scores(axarr[1], k_gen_est_errors, "K-Fold method", "Test")

    f.text(0.5, 0.04, 'Model index', ha='center', va='center')
    f.text(0.06, 0.5, 'Generalisation error', ha='center', va='center', rotation='vertical')

    plot.show()

if __name__ == "__main__":
    main()
