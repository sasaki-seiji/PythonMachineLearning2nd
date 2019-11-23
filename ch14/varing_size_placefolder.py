# Placeholder for varying batchsizes:
import tensorflow as tf
import numpy as np



g = tf.Graph()

with g.as_default():
    tf_x = tf.compat.v1.placeholder(tf.float32, 
                          shape=[None, 2],
                          name='tf_x')
    
    x_mean = tf.reduce_mean(tf_x, 
                          axis=0, 
                          name='mean')


np.random.seed(123)
np.set_printoptions(precision=2)

with tf.compat.v1.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, 
                           size=(5,2))
    print('Feeding data with shape', x1.shape)
    print('Result:', sess.run(x_mean, 
                             feed_dict={tf_x:x1}))
    x2 = np.random.uniform(low=0, high=1, 
                           size=(10,2))
    print('Feeding data with shape', x2.shape)
    print('Result:', sess.run(x_mean, 
                             feed_dict={tf_x:x2}))




print(tf_x)

