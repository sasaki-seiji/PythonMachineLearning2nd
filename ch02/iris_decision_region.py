import matplotlib.pyplot as plt
from iris_perceptron import ppn, X, y
from decision_region import plot_decision_regions

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()
