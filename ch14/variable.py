import tensorflow as tf
import numpy as np
# #### Defining Variables


g1 = tf.Graph()

with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]), name='w')
    print(w)

# #### Initializing variables



## initialize w and evaluate it
with tf.compat.v1.Session(graph=g1) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(w))




## add the init_op to the graph
with g1.as_default():
    init_op = tf.compat.v1.global_variables_initializer()
    
## initialize w with init_op and evaluate it
with tf.compat.v1.Session(graph=g1) as sess:
    sess.run(init_op)
    print(sess.run(w))




g2 = tf.Graph()

with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.compat.v1.global_variables_initializer()
    w2 = tf.Variable(2, name='w2')




with tf.compat.v1.Session(graph=g2) as sess:
    sess.run(init_op)
    print('w1:', sess.run(w1))


# Error if a variable is not initialized:



with tf.compat.v1.Session(graph=g2) as sess:
    
    try:
        sess.run(init_op)
        print('w2:', sess.run(w2))
    except tf.errors.FailedPreconditionError as e:
        print(e)


