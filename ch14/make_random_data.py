## create a random toy dataset for regression
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, 
                             scale=(0.5 + t*t/3), 
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)


x, y = make_random_data() 

# 2019.11.24 change
if __name__ == '__main__':
    plt.plot(x, y, 'o')
    # plt.savefig('images/14_03.png', dpi=300)
    plt.show()
