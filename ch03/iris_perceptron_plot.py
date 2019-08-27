
import numpy as np
import matplotlib.pyplot as plt

from decision_region import plot_decision_regions
from iris_split import y_train, y_test
from iris_std import X_train_std, X_test_std
from iris_perceptron import ppn

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()
