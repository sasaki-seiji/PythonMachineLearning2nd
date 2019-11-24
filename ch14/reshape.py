import tensorflow as tf

from shape import g, T1

with g.as_default():
    T4 = tf.reshape(T1, shape=[1, 1, -1], 
                    name='T4')
    print(T4)
    T5 = tf.reshape(T1, shape=[1, 3, -1], 
                    name='T5')
    print(T5)




with tf.compat.v1.Session(graph = g) as sess:
    print(sess.run(T4)) 
    print()   
    print(sess.run(T5))

