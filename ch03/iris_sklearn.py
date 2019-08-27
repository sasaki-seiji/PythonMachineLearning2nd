from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

if __name__ == '__main__':
    print('Class labels:', np.unique(y))
