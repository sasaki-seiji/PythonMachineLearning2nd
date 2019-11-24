## TensorFlow control flow
import tensorflow as tf


x, y = 1.0, 2.0

g = tf.Graph()
with g.as_default():
    tf_x = tf.compat.v1.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_x')
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_y')
    res = tf.cond(tf_x < tf_y, 
                  lambda: tf.add(tf_x, tf_y, 
                                 name='result_add'),
                  lambda: tf.subtract(tf_x, tf_y, 
                                 name='result_sub'))
    print('Object:', res)
        
with tf.compat.v1.Session(graph=g) as sess:
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x, 
                              'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x,
                              'tf_y:0': y}))  

    file_writer = tf.compat.v1.summary.FileWriter(logdir='./logs/tf-cond/', graph=g)
