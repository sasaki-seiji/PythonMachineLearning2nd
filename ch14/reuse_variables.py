import tensorflow as tf

from helper_functions import build_classifier, build_generator
########################
## Defining the graph ##
########################

batch_size=64
g = tf.Graph()

with g.as_default():
    tf_X = tf.compat.v1.placeholder(shape=(batch_size, 100), 
                          dtype=tf.float32,
                          name='tf_X')
    ## build the generator
    with tf.compat.v1.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, 
                                   n_hidden=50)
    
    ## build the classifier
    with tf.compat.v1.variable_scope('classifier') as scope:
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, 
                                    labels=tf.ones(
                                        shape=batch_size))
        
        ## reuse the classifier for generated data
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(
                                        shape=batch_size))
        
        init_op = tf.compat.v1.global_variables_initializer()

