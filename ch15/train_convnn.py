from std_mnist import X_train_centered, y_train, X_valid_centered, y_valid
from convnn import ConvNN

cnn = ConvNN(random_seed=123)
#cnn = ConvNN(epochs=2, random_seed=123)

cnn.train(training_set=(X_train_centered, y_train), 
          validation_set=(X_valid_centered, y_valid))

cnn.save(epoch=20)
#cnn.save(epoch=2)
