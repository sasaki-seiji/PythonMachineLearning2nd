# ### Executing objects in a TensorFlow graph using their names
import matplotlib.pyplot as plt
import tensorflow as tf

from regression_model import g, tf_x, tf_y, cost, train_op
from make_random_data import x, y

## train/test splits:
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]


## training the model
n_epochs = 500
training_costs = []
with tf.compat.v1.Session(graph=g) as sess:
    ## first, run the variables initializer
    sess.run(tf.compat.v1.global_variables_initializer())
    
    ## train the model for n_epochs
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], 
                        feed_dict={tf_x: x_train,
                                   tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))
            

plt.plot(training_costs)
# plt.savefig('images/14_04.png', dpi=300)

# 2019.11.24 add
plt.show()

