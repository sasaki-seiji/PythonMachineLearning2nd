# ### First steps with TensorFlow
import tensorflow as tf




## create a graph
g = tf.Graph()
with g.as_default():
# 2019.11.03 changed:    
#    x = tf.placeholder(dtype=tf.float32,
#                       shape=(None), name='x')
    x = tf.compat.v1.placeholder(dtype=tf.float32,
                       shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')

    z = w*x + b
# 2019.11.03 change
    #init = tf.global_variables_initializer()
    init = tf.compat.v1.global_variables_initializer()

## create a session and pass in graph g
# 2019.11.03 change
with tf.compat.v1.Session(graph=g) as sess:
    ## initialize w and b:
    sess.run(init)
    ## evaluate z:
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f'%(
              t, sess.run(z, feed_dict={x:t})))




with tf.compat.v1.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x:[1., 2., 3.]})) 

