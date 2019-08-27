from sklearn.preprocessing import StandardScaler
from iris_split import X_train, X_test

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

if __name__ == '__main__':
    print('X_train_std[0:3]:\n', X_train_std[:3]) 
    print('X_test_std[0:3]:\n', X_test_std[:3])               