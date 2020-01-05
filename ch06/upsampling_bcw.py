import numpy as np
from sklearn.utils import resample

from imbalance_bcw import X, y, X_imb, y_imb


print('Number of class 1 samples before:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

print('Number of class 1 samples after:', X_upsampled.shape[0])




X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))




y_pred = np.zeros(y_bal.shape[0])
# 2020.01.05 change
#np.mean(y_pred == y_bal) * 100
acc = np.mean(y_pred == y_bal) * 100
print('accuracy: ', acc)