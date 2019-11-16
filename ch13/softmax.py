# ### Estimating class probabilities in multi-class classification via the softmax function
import numpy as np

from logistic_output import Z

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)




np.sum(y_probas)
# 2019.11.16 add
print('sum: ', np.sum(y_probas))
