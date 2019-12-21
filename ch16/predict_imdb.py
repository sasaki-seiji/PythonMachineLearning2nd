import numpy as np

from prepare_imdb import word_to_int, sequence_length, X_train, y_train, X_test, y_test
from sentiment_rnn import SentimentRNN

## Train:

n_words = max(list(word_to_int.values())) + 1

rnn = SentimentRNN(n_words=n_words, 
                   seq_len=sequence_length,
                   embed_size=256, 
                   lstm_size=128, 
                   num_layers=1, 
                   batch_size=100, 
                   learning_rate=0.001)




rnn.train(X_train, y_train, num_epochs=40)




## Test: 
preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: %.3f' % (
      np.sum(preds == y_true) / len(y_true)))




## Get probabilities:
proba = rnn.predict(X_test, return_proba=True)
