'''
Created on Jul 5, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU, time_distributed_dense
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.engine.topology import merge
import numpy as np

import logging

logger = logging.getLogger(__name__)

def check_and_throw_if_fail(condition, msg):
    '''
    condition: boolean; if condition is False, log and throw an exception
    msg: the message to log and exception if condition is False
    '''
    if not condition:
        logger.error(msg)
        raise Exception(msg)

class Attention(Layer):
    '''
    Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
    '''
    def __init__(self, attention_weight_vector_dim, element_wise_output_transformer = None, **kwargs):
        '''
        attention_weight_vector_dim: dimension of the attention weight vector 
        element_wise_output_transformer: element-wise output transformer,e.g., K.sigmoid
        '''
        check_and_throw_if_fail(attention_weight_vector_dim > 0 , "attention_weight_vector_dim")
        self.attention_weight_vector_dim = attention_weight_vector_dim
        self.element_wise_output_transformer = element_wise_output_transformer
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        input_dim = input_shape[2]
        # TODO: better way to initialize parameters
        initial_Ws = np.random.random((input_dim, self.attention_weight_vector_dim))
        initial_bs = np.random.random((self.attention_weight_vector_dim,))
        initial_us = np.random.random((self.attention_weight_vector_dim,))
        self.Ws = K.variable(initial_Ws)
        self.bs = K.variable(initial_bs)
        self.us = K.variable(initial_us)
        self.trainable_weights = [self.Ws, self.bs, self.us]

    def call(self, x, mask = None):
        '''
        x: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")

        input_dim = K.int_shape(x)[2]
        time_steps = K.int_shape(x)[1]
        
        ui = K.tanh(time_distributed_dense(x, self.Ws, self.bs,input_dim=input_dim,output_dim=self.attention_weight_vector_dim,timesteps=time_steps))  # batch_size, time_steps, attention_weight_vector_dim
        ai = K.exp(time_distributed_dense(ui, K.expand_dims(self.us,1),input_dim=self.attention_weight_vector_dim,output_dim=1,timesteps=time_steps))  # batch_size, time_steps, 1
        sum_of_ai = K.sum(ai, 1, keepdims = True)  # batch_size 1 1
        
        sum_of_ai = K.repeat_elements(sum_of_ai, rep = time_steps, axis = 1)  # batch_size time_steps 1
        ai = ai / sum_of_ai  # batch_size * time_steps * 1
        
        ai = K.repeat_elements(ai, rep = input_dim, axis = 2)  # batch_size time_steps input_dim
        # batch_size *time_steps * input_dim -> batch_size* input_dim
        output = K.sum(ai * x, 1)
        if self.element_wise_output_transformer:
            return self.element_wise_output_transformer(output)
        else:
            return output

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        return (input_shape[0], input_shape[2])

class SequenceToSequenceEncoder(Layer):
    def __init__(self, output_dim, is_bi_directional  = True, **kwargs):        
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        self.output_dim = output_dim
        self.is_bi_directional = is_bi_directional
        super(SequenceToSequenceEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        self.encoder_left_to_right = GRU(self.output_dim, return_sequences = True)
        if self.is_bi_directional:
            self.encoder_right_to_left = GRU(self.output_dim, return_sequences = True, go_backwards = True)
        
    def call(self, x, mask = None):
        '''
        x: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        h1 =self. encoder_left_to_right(x)
        if self.is_bi_directional:
            h2 = self.encoder_right_to_left(x)
            return K.concatenate([h1, h2], 2)
        else:
            return h1

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        if self.is_bi_directional:
            return  input_shape[:-1] + (2*self.output_dim,)
        else:
            return  input_shape[:-1] + (self.output_dim,)

class SequenceToVectorEncoder(Layer):
    def __init__(self, output_dim, is_bi_directional  = True, **kwargs):        
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        self.output_dim = output_dim
        super(SequenceToVectorEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        self.conv = Convolution1D(self.output_dim, 3, border_mode = 'same')
        timesteps = input_shape[1]
        self.pooling = MaxPooling1D(pool_length = timesteps)
        
    def call(self, x, mask = None):
        '''
        x: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        output = self.conv(x)
        output = self.pooling(output)  # batch_size * 1 * output_dim
        # to remove the time step dimension
        return K.squeeze(output, 1)

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        return (input_shape[0],self.output_dim)
    

def apply_attention_layer_with_sequence_to_sequence_encoder(input_sequence, output_dim, attention_weight_vector_dim, element_wise_output_transformer = None):
    '''
    input_sequence: input sequence, batch_size*time_steps*input_dim
    output_dim: dimension of output vector
    attention_weight_vector_dim: dimension of attention weight vector
    element_wise_output_transformer: element wise output transformer
    '''
    check_and_throw_if_fail(K.ndim(input_sequence) == 3 , "input_sequence")
    attention_layer = Attention(attention_weight_vector_dim, element_wise_output_transformer)
    sequence_to_sequence_encoder= SequenceToSequenceEncoder(output_dim)
    transformed_sequence = sequence_to_sequence_encoder(input_sequence)
    return attention_layer(transformed_sequence)

def apply_attention_layer_with_sequence_to_vector_encoder(input_sequence, output_dim, attention_weight_vector_dim, element_wise_output_transformer = None):
    '''
    input_sequence: input sequence, batch_size*time_steps*input_dim
    sequence_to_vector_encoder: transform the input sequence into a vector
    attention_weight_vector_dim: dimension of attention weight vector
    element_wise_output_transformer: element wise output transformer
    '''
    check_and_throw_if_fail(K.ndim(input_sequence) == 3 , "input_sequence")
    check_and_throw_if_fail(output_dim > 0 , "output_dim")
    attention_layer = Attention(attention_weight_vector_dim, element_wise_output_transformer)
    sequence_to_vector_encoder=SequenceToVectorEncoder(output_dim)
    transformed_vector = sequence_to_vector_encoder(input_sequence)
    attention_vector = attention_layer(input_sequence)
    return merge(inputs=[attention_vector, transformed_vector], mode='concat', concat_axis=-1)
