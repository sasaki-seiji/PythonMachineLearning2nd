import numpy as np

from prepare_hamlet_text import text_ints, int2char

def reshape_data(sequence, batch_size, num_steps):
    mini_batch_length = batch_size * num_steps
    num_batches = int(len(sequence) / mini_batch_length)
    if num_batches*mini_batch_length + 1 > len(sequence):
        num_batches = num_batches - 1
    ## Truncate the sequence at the end to get rid of 
    ## remaining charcaters that do not make a full batch
    x = sequence[0 : num_batches*mini_batch_length]
    y = sequence[1 : num_batches*mini_batch_length + 1]
    ## Split x & y into a list batches of sequences: 
    x_batch_splits = np.split(x, batch_size)
    y_batch_splits = np.split(y, batch_size)
    ## Stack the batches together
    ## batch_size x mini_batch_length
    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)
    
    return x, y

## Testing:
train_x, train_y = reshape_data(text_ints, 64, 10)
print(train_x.shape)
print(train_x[0, :10])
print(train_y[0, :10])
print(''.join(int2char[i] for i in train_x[0, :50]))
