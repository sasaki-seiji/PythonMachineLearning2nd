from sklearn.preprocessing import StandardScaler
from wine_dataset import X_train, X_test

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# 2019.09.10 add
if __name__ == '__main__':
    print(X_train_std[:2, :])
    print(X_test_std[:2, :])