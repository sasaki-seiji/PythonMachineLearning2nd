from iris_split import y_test
from iris_std import X_test_std
from iris_perceptron import ppn

y_pred = ppn.predict(X_test_std)

if __name__ == '__main__':
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
