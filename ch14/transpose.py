import tensorflow as tf

from reshape import g, T5

with g.as_default():
    T6 = tf.transpose(T5, perm=[2, 1, 0], 
                     name='T6')
    print(T6)
    T7 = tf.transpose(T5, perm=[0, 2, 1], 
                     name='T7')
    print(T7)

# 2019.11.24 add
with tf.compat.v1.Session(graph = g) as sess:
    print(sess.run(T6)) 
    print()   
    print(sess.run(T7))

