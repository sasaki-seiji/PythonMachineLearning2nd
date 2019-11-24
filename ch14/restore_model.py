# Restoring the saved model:
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from make_random_data import x, y

# 2019.11.24 add
## train/test splits
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]

## new file: loading a trained model
## and run the model on test set


g2 = tf.Graph()
with tf.compat.v1.Session(graph=g2) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_pred = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_test})




print('SSE: %.4f' % (np.sum(np.square(y_pred - y_test))))




x_arr = np.arange(-2, 4, 0.1)

g2 = tf.Graph()
with tf.compat.v1.Session(graph=g2) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_arr = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_arr})

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr.T[:, 0], '-r', lw=3)
# plt.savefig('images/14_05.png', dpi=400)
plt.show()
