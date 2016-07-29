'''
Created on Jul 13, 2016

@author: lxh5147
'''

from attention_layer import shape
from keras import backend as K
import numpy as np

def fake_data(input_shape, dtype='float32', max_int=10):
    random_shape = []
    for shape in input_shape:
        if shape is None:
            n = int(np.random.random() * 10)
            if n == 0 : 
                n = 1
            random_shape.append(n)
        else:
            random_shape.append(shape)
    val = np.random.random(random_shape)
    if dtype == 'int32':
        val = val * max_int
        val = val.astype(int)
    return val

def to_binary_matrix(x, max_int):
    '''
    x: a list of positions, whose value is not zero
    y: one hot representation
    '''
    len_x = len(x)
    y = np.zeros((len_x, max_int))
    y[np.arange(len_x), x] = 1
    return y

def faked_dataset(inputs, total, timesteps, vocabulary_size, output_dim):
    input_vals = []
    for input_tensor in inputs:
        input_val = fake_data((total,) + shape(input_tensor)[1:], K.dtype(input_tensor), max_int=vocabulary_size)
        input_vals.append(input_val)
    output_val = fake_data((total, timesteps) , 'int32', max_int=output_dim)
    output_val = output_val.reshape((total * timesteps,))
    output_val = to_binary_matrix(output_val, max_int=output_dim)
    output_val = output_val.reshape((total, timesteps, output_dim))
    return input_vals, output_val


