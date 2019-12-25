from prepare_hamlet_text import text_ints, chars
from char_rnn_utiles import reshape_data
from char_rnn import CharRNN

batch_size = 64
num_steps = 100 
train_x, train_y = reshape_data(text_ints, 
                                batch_size, 
                                num_steps)

rnn = CharRNN(num_classes=len(chars), batch_size=batch_size)
rnn.train(train_x, train_y, 
          num_epochs=100,
          ckpt_dir='./model-100/')
