from sklearn.svm import SVC
import matplotlib.pyplot as plt

from decision_region import plot_decision_regions
from iris_split import y_test, y_train
from iris_std import X_train_std
from iris_combined import X_combined_std, y_combined

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()
