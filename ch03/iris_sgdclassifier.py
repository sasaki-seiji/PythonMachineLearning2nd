from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

from decision_region import plot_decision_regions
from iris_split import y_train
from iris_std import X_train_std
from iris_combined import X_combined_std, y_combined

# 2019.08.31: chage n_iter param -> max_iter
#ppn = SGDClassifier(loss='perceptron', n_iter=1000)
#lr = SGDClassifier(loss='log', n_iter=1000)
#svm = SGDClassifier(loss='hinge', n_iter=1000)
ppn = SGDClassifier(loss='perceptron', max_iter=1000)
lr = SGDClassifier(loss='log', max_iter=1000)
svm = SGDClassifier(loss='hinge', max_iter=1000)

# peceptron
ppn.fit(X_train_std, y_train)
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.title('perceptron')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()

# logistic regression
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.title('logistic regression')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()

# support vector machine
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.title('SVM')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()
