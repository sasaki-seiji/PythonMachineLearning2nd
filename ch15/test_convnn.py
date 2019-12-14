import numpy as np

from std_mnist import X_test_centered, y_test
from convnn import ConvNN

cnn = ConvNN(random_seed=123)

cnn.load(epoch=20, path='./tflayers-model/')

print(cnn.predict(X_test_centered[:10,:]))

preds = cnn.predict(X_test_centered)

print('Test Accuracy: %.2f%%' % (100*
      np.sum(y_test == preds)/len(y_test)))

