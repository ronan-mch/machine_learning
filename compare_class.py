import titanic_data
import forward_select as fwd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    linear = titanic_data.linear()
    data = linear[:-89]
    gen_est_data = linear[-89:]
    tree_baseline = fwd.get_baseline(data, test_func=fwd.score_tree, target='Survived')
    predictor_features = data.columns.drop('Survived')

    # Decision Tree
    tree_best, tree_scorechart = fwd.forward_select(data, [], predictor_features, tree_baseline, fwd.score_cnn, scores=[], target='Survived')
    tree_labels, tree_scores, tree_error_rates = fwd.process_scorechart(tree_scorechart)
    tree_gen_est_errors = fwd.estimate_generalisation_errors(tree_scorechart, data, gen_est_data, fwd.score_cnn, 'Survived')
    print "Decision Tree"
    fwd.table_display(tree_labels, tree_scores, tree_gen_est_errors)

    # Neural Network
    nn_baseline = fwd.get_baseline(data, test_func=fwd.score_cnn, target='Survived')
    nn_baseline = 1
    nn_best, nn_scorechart = fwd.forward_select(data, [], predictor_features, nn_baseline, test_func=fwd.score_nn, est_func=fwd.test_kfold, scores=[], target='Survived')
    nn_labels, nn_scores, nn_error_rates = fwd.process_scorechart(nn_scorechart)
    nn_gen_est_errors = fwd.estimate_generalisation_errors(nn_scorechart, data, gen_est_data, fwd.score_nn, 'Survived')
    print "Neural Network"
    fwd.table_display(nn_labels, nn_scores, nn_gen_est_errors)

    # K-Neighbours
    knn_best_model, knn_scorechart = fwd.forward_select(data, [], data.columns.drop('Survived'), 1, fwd.score_knn, target='Survived')
    knn_gen_est_errors = fwd.estimate_generalisation_errors(knn_scorechart, data, gen_est_data, fwd.score_knn, 'Survived')
    knn_labels, knn_scores, knn_errors = fwd.process_scorechart(knn_scorechart)
    print "K-Neighbour"
    fwd.table_display(knn_labels, knn_scores, knn_gen_est_errors)

    # Assume largest class - largest class did not survive (0)
    all_died = np.zeros(np.shape(gen_est_data.Survived.values))
    all_died_error = np.mean((all_died - gen_est_data.Survived.values) ** 2)

    # Graphing code
    f, axarr = plot.subplots(1, 3)

    fwd.plot_scores(axarr[0], tree_scores, "Decision Tree", "Train")
    fwd.plot_scores(axarr[0], tree_gen_est_errors, "Decision Tree", "Test")
    all_died_arr = np.empty(len(tree_scores))
    all_died_arr.fill(all_died_error)
    axarr[0].plot(all_died_arr, label="Assume Largest Class")
    axarr[0].legend()
    plot.yticks([0, .25, .5, .75, 1])

    fwd.plot_scores(axarr[1], nn_scores, "Neural Network", "Train")
    fwd.plot_scores(axarr[1], nn_gen_est_errors, "Neural Network", "Test")
    all_died_arr = np.empty(len(nn_scores))
    all_died_arr.fill(all_died_error)
    axarr[1].plot(all_died_arr, label="Assume Largest Class")
    axarr[1].legend()
    plot.yticks([0, .25, .5, .75, 1])

    fwd.plot_scores(axarr[2], knn_scores, "K-Neighbors", "Train")
    fwd.plot_scores(axarr[2], knn_gen_est_errors, "K-Neighbors", "Test")
    all_died_arr = np.empty(len(knn_scores))
    all_died_arr.fill(all_died_error)
    axarr[2].plot(all_died_arr, label="Assume Largest Class")
    axarr[2].legend()
    plot.yticks([0, .25, .5, .75, 1])

    f.text(0.5, 0.04, 'Model index', ha='center', va='center')

    f.text(0.06, 0.5, 'Generalisation error', ha='center', va='center', rotation='vertical')

    plot.show()

if __name__ == "__main__":
    main()
