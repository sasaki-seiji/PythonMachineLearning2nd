from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from decision_region import plot_decision_regions
from iris_split import y_train
from iris_std import X_train_std
from iris_combined import X_combined_std, y_combined

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()
