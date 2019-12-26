import numpy as np

from prepare_hamlet_text import chars
from char_rnn import CharRNN

np.random.seed(123)
rnn = CharRNN(len(chars), sampling=True)

print(rnn.sample(ckpt_dir='./model-100/', 
                 output_length=500))
