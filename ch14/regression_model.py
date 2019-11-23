# ### Building a regression model
import tensorflow as tf



## define a graph
g = tf.Graph()

## define the computation graph
with g.as_default():
    ## placeholders
    tf.compat.v1.set_random_seed(123)
    tf_x = tf.compat.v1.placeholder(shape=(None), 
                          dtype=tf.float32, 
                          name='tf_x')
    tf_y = tf.compat.v1.placeholder(shape=(None), 
                          dtype=tf.float32,
                          name='tf_y')
    
    ## define the variable (model parameters)
    weight = tf.Variable(
        tf.compat.v1.random_normal(
            shape=(1, 1), 
            stddev=0.25),
        name='weight')
    bias = tf.Variable(0.0, name='bias')
    
    ## build the model
    y_hat = tf.add(weight * tf_x, bias, 
                   name='y_hat')
    print(y_hat)
    
    ## compute the cost
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), 
                          name='cost')
    print(cost)
    
    ## train
    optim = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')

