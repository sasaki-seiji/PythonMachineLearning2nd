import numpy as np
from lda_eigenvalues_wine_data import eigen_pairs

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
