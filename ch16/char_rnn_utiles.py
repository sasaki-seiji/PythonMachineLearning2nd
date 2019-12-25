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


np.random.seed(123)

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, tot_batch_length = data_x.shape    
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:, b*num_steps: (b+1)*num_steps], 
               data_y[:, b*num_steps: (b+1)*num_steps])
        
bgen = create_batch_generator(train_x[:,:100], train_y[:,:100], 15)
for b in bgen:
    print(b[0].shape, b[1].shape, end='  ')
    print(''.join(int2char[i] for i in b[0][0,:]).replace('\n', '*'), '    ',
          ''.join(int2char[i] for i in b[1][0,:]).replace('\n', '*'))


def get_top_char(probas, char_size, top_n=5):
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p / np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id
