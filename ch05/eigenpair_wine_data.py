import numpy as np

from std_wine_data import X_train_std

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

# 2019.12.26 add
print('\nEigenvectors \n%s' % eigen_vecs)