import tensorflow as tf

from reshape import g, T5

with g.as_default():
    t5_splt = tf.split(T5, 
                       num_or_size_splits=2, 
                       axis=2, name='T8')
    print(t5_splt)

# 2019.11.24 add
with tf.compat.v1.Session(graph = g) as sess:
    print(sess.run(t5_splt)) 
