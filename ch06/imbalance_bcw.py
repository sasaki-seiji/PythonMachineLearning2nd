import numpy as np

from prepare_breast_cancer import X, y

X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))




y_pred = np.zeros(y_imb.shape[0])
# 2020.01.05 change
#np.mean(y_pred == y_imb) * 100
acc = np.mean(y_pred == y_imb) * 100
print('accuracy: ', acc)