import numpy as np

from load_mnist import load_mnist

## loading the data
#X_train, y_train = load_mnist('.', kind='train')
X_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d,  Columns: %d' %(X_train.shape[0], 
                                 X_train.shape[1]))

#X_test, y_test = load_mnist('.', kind='t10k')
X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d,  Columns: %d' %(X_test.shape[0], 
                                     X_test.shape[1]))
## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)

print(X_test_centered.shape, y_test.shape)

