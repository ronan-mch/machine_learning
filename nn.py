from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import titanic_data
import forward_select as fwd

data = titanic_data.linear()
train = data[:-50]
gen_test = data[-50:]
best_model, scorechart = fwd.forward_select(train, [], data.columns.drop('Survived'), 1, fwd.score_cnn, target='Survived')

gen_est_errors = fwd.estimate_generalisation_errors(scorechart, train, gen_test, fwd.score_cnn, 'Survived')
labels, scores, errors = fwd.process_scorechart(scorechart)
plt.plot(scores, label="Training error", marker='o')
plt.plot(gen_est_errors, label="Test error", marker='o')
plt.title("Neural Network")
plt.xlabel('Model index')
plt.ylabel('Generalisation error')
plt.xticks(range(len(scores)), range(1, len(scores) + 1))
plt.yticks([0, 0.5, 1])
plt.legend()
plt.show()
