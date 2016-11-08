import titanic_data
import forward_select as fwd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt

data = titanic_data.linear()
train = data[:-50]
gen_test = data[-50:]
knn_best_model, knn_scorechart = fwd.forward_select(train, [], data.columns.drop('Survived'), 1, fwd.score_knn, target='Survived')
knn_gen_est_errors = fwd.estimate_generalisation_errors(knn_scorechart, train, gen_test, fwd.score_knn, 'Survived')
knn_labels, knn_scores, knn_errors = fwd.process_scorechart(knn_scorechart)
plt.plot(knn_scores, label="Training error", marker='o')
plt.plot(knn_gen_est_errors, label="Test error", marker='o')
plt.title("K-Neighbours")
plt.xlabel('Model index')
plt.ylabel('Generalisation error')
plt.xticks(range(len(knn_scores)), range(1, len(knn_scores) + 1))
plt.yticks([0, 0.5, 1])
plt.legend()
plt.show()
