from sklearn.metrics import accuracy_score

from iris_split import y_test
from iris_perceptron_predict import y_pred

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
