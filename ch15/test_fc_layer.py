import tensorflow as tf

from fc_layer import fc_layer

## testing:
g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(tf.float32, 
                       shape=[None, 28, 28, 1])
    fc_layer(x, name='fctest', n_output_units=32, 
             activation_fn=tf.nn.relu)
    
del g, x
