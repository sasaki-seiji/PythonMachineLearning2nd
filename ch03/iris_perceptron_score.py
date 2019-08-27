from iris_perceptron import ppn
from iris_split import y_test
from iris_std import X_test_std

print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))
