from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from decision_region import plot_decision_regions
from iris_split import y_train
from iris_std import X_train_std
from iris_combined import X_combined_std, y_combined

knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()

