import numpy as np

from std_wine_data import X_train_std, y_train
from class_means_wine_data import mean_vecs

print('Class label distribution: %s' 
      % np.bincount(y_train)[1:])




d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                     S_W.shape[1]))
