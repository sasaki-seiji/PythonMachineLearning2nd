import pickle
import os

from learn_neuralnet_mnist import nn

pickle.dump(nn, open('neuralnet_mnist.pkl', 'wb'), protocol=4)