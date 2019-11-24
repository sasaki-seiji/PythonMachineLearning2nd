## Python control flow
import tensorflow as tf


x, y = 1.0, 2.0

g = tf.Graph()
with g.as_default():
    tf_x = tf.compat.v1.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_x')
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, 
                           shape=None, name='tf_y')
    if x < y:
        res = tf.add(tf_x, tf_y, name='result_add')
    else:
        res = tf.subtract(tf_x, tf_y, name='result_sub')
        
    print('Object: ', res)
        
with tf.compat.v1.Session(graph=g) as sess:
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x, 
                              'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: %s -> Result:' % (x < y), 
          res.eval(feed_dict={'tf_x:0': x,
                              'tf_y:0': y}))  
    
    ## uncomment the next line if you want to visualize the graph in TensorBoard:
    file_writer = tf.compat.v1.summary.FileWriter(logdir='./logs/py-cflow/', graph=g)
