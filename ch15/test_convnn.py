import numpy as np

from std_mnist import X_test_centered, y_test
from train_convnn import cnn

print(cnn.predict(X_test_centered[:10,:]))

preds = cnn.predict(X_test_centered)

print('Test Accuracy: %.2f%%' % (100*
      np.sum(y_test == preds)/len(y_test)))

