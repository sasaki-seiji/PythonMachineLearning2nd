from sklearn.preprocessing import MinMaxScaler
from wine_dataset import X_train, X_test

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# 2019.09.10 add
if __name__ == '__main__':
    print(X_train_norm[:2, :])
    print(X_test_norm[:2, :])