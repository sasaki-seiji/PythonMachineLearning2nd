import tensorflow as tf
import numpy as np

from build_cnn import build_cnn
from cnn_utils import train, save
from std_mnist import X_train_centered, y_train, X_valid_centered, y_valid

## Define random seed
random_seed = 123

np.random.seed(random_seed)


## create a graph
g = tf.Graph()
with g.as_default():
    tf.compat.v1.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

    ## saver:
    saver = tf.compat.v1.train.Saver()

## crearte a TF session 
## and train the CNN model

with tf.compat.v1.Session(graph=g) as sess:
    train(sess, 
          training_set=(X_train_centered, y_train), 
          validation_set=(X_valid_centered, y_valid), 
          initialize=True,
          random_seed=123)
    save(saver, sess, epoch=20)

