import numpy as np

from iris_split import y_train, y_test
from iris_std import X_train_std, X_test_std

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
