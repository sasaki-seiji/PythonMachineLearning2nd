# #### Defining placeholders
import tensorflow as tf



g = tf.Graph()
with g.as_default():
    tf_a = tf.compat.v1.placeholder(tf.int32, shape=[],
                          name='tf_a')
    tf_b = tf.compat.v1.placeholder(tf.int32, shape=[],
                          name='tf_b') 
    tf_c = tf.compat.v1.placeholder(tf.int32, shape=[],
                          name='tf_c') 

    r1 = tf_a-tf_b
    r2 = 2*r1
    z  = r2 + tf_c


# #### Feeding placeholders with data



## launch the previous graph
with tf.compat.v1.Session(graph=g) as sess:
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('z:', 
          sess.run(z, feed_dict=feed))


# Execution with and without feeding tf_c:



## launch the previous graph
with tf.compat.v1.Session(graph=g) as sess:
    ## execution without feeding tf_c
    feed = {tf_a: 1,
            tf_b: 2}
    print('r1:', 
          sess.run(r1, feed_dict=feed))
    print('r2:', 
          sess.run(r2, feed_dict=feed))
    
    ## execution with feeding tf_c
    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('r1:', 
          sess.run(r1, feed_dict=feed))
    print('r2:', 
          sess.run(r2, feed_dict=feed))
