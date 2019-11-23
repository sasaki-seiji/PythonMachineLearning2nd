# #### Variable scope
import tensorflow as tf



g = tf.Graph()

with g.as_default():
    with tf.compat.v1.variable_scope('net_A'):
        with tf.compat.v1.variable_scope('layer-1'):
            w1 = tf.Variable(tf.compat.v1.random_normal(
                shape=(10,4)), name='weights')
        with tf.compat.v1.variable_scope('layer-2'):
            w2 = tf.Variable(tf.compat.v1.random_normal(
                shape=(20,10)), name='weights')
    with tf.compat.v1.variable_scope('net_B'):
        with tf.compat.v1.variable_scope('layer-1'):
            w3 = tf.Variable(tf.compat.v1.random_normal(
                shape=(10,4)), name='weights')

    print(w1)
    print(w2)
    print(w3)

