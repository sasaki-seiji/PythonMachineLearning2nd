from sentiment_rnn import SentimentRNN

#n_words = max(list(word_to_int.values())) + 1
n_words = 10000 + 1
sequence_length = 200

rnn = SentimentRNN(n_words=n_words, 
                   seq_len=sequence_length,
                   embed_size=256, 
                   lstm_size=128, 
                   num_layers=1, 
                   batch_size=100, 
                   learning_rate=0.001)
