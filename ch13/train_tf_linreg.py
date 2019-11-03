import tensorflow as tf
import numpy as np

from tf_linreg import TfLinreg, train_linreg

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3, 
                    6.6, 7.4, 8.0, 
                    9.0])

lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)

sess = tf.compat.v1.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)

# 2019.11.04 add
if __name__ == '__main__':
    print(training_costs)