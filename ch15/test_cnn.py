### Calculate prediction accuracy
### on test set
### restoring the saved model
import tensorflow as tf
import numpy as np

from build_cnn import build_cnn
from cnn_utils import load, predict
from std_mnist import X_test_centered, y_test

# 2019.12.07 add
## Define random seed
random_seed = 123
np.random.seed(random_seed)

## create a new graph 
## and build the model
g2 = tf.Graph()
with g2.as_default():
    tf.compat.v1.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

    ## saver:
    saver = tf.compat.v1.train.Saver()

## create a new session 
## and restore the model
with tf.compat.v1.Session(graph=g2) as sess:
    load(saver, sess, 
         epoch=20, path='./model/')
    
    preds = predict(sess, X_test_centered, 
                    return_proba=False)

    print('Test Accuracy: %.3f%%' % (100*
                np.sum(preds == y_test)/len(y_test)))
    
