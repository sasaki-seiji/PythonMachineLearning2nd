from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =       train_test_split(X, y, 
                        test_size=0.5, 
                        random_state=1,
                        stratify=y)

# 2020.01.07 add
print('X_train[0]: ', X_train[0], ', y_train[0]: ', y_train[0])