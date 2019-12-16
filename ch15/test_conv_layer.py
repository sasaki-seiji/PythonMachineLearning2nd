import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from conv_layer import conv_layer

## testing
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name='convtest', kernel_size=(3, 3), n_output_channels=32)
    
del g, x
