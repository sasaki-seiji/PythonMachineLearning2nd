import tensorflow as tf
import numpy as np

from build_cnn import build_cnn

## Define random seed
random_seed = 123

np.random.seed(random_seed)


## create a graph
g = tf.Graph()
with g.as_default():
    tf.compat.v1.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

with tf.compat.v1.Session(graph=g) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    file_write = tf.compat.v1.summary.FileWriter(logdir='./logs/', graph=g)