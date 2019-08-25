import matplotlib.pyplot as plt

from perceptron import Perceptron
from iris_array import X, y

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

if __name__ == '__main__':
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    # plt.savefig('images/02_07.png', dpi=300)
    plt.show()