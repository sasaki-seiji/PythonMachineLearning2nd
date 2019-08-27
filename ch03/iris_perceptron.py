from sklearn.linear_model import Perceptron
from iris_split import y_train
from iris_std import X_train_std

# 2019.08.27 change: n_iter to max_iter
#ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

if __name__ == '__main__':
    print(ppn)
