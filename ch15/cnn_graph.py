import numpy as np
# 2019.12.18 change
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from build_cnn import build_cnn

## Define random seed
random_seed = 123

np.random.seed(random_seed)


## create a graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    file_write = tf.summary.FileWriter(logdir='./logs/', graph=g)