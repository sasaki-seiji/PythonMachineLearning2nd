import numpy as np
from sklearn.linear_model import LogisticRegression

from wine_dataset import y_train, y_test
from wine_std import X_train_std, X_test_std

lr = LogisticRegression(penalty='l1', C=1.0)
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))




lr.intercept_
# 2019.09.17 add
if __name__ == '__main__':
    print(lr.intercept_)




np.set_printoptions(8)




lr.coef_[lr.coef_!=0].shape
# 2019.09.17 add
if __name__ == '__main__':
    print(lr.coef_[lr.coef_!=0].shape)



lr.coef_
# 2019.09.17 add
if __name__ == '__main__':
    print(lr.coef_)

