'''
Created on Jul 5, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU, time_distributed_dense
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Input
from keras.layers.embeddings import Embedding

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
    '''
    Represents an encoder that transforms input sequence to another sequence
    '''
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
        returns a tensor of shape batch_size * time_steps * 2*input_dim (or input_dim if not bidirectional)  
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
    '''
    Represents an encoder that transforms a sequence into a vector 
    '''
    def __init__(self, output_dim, **kwargs):        
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
        Returns a tensor of the shape: batch_size * output_dim
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
    
class HierarchicalAttention(Layer):
    '''
    Represents a hierarchical attention layer
    One example: snapshots* documents * sections* sentences * words
    '''
    def __init__(self, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding=None, use_sequence_to_vector_encoder = False,  **kwargs):
        '''
        top_feature_dim: dim of the top feature, e.g., the snapshot level feature
        attention_output_dims: attention output dimensions on different levels: e.g., section, document, sentence, word
        attention_weight_vector_dims: weight vector dimensions inside each attention layer, e.g., section, document, sentence, word
        use_sequence_to_vector_encoder: True if use sequence to vector encoder otherwise sequence to sequence encoder inside all attention layers
        '''
        check_and_throw_if_fail(len(attention_output_dims) > 0 , "attention_output_dims")
        check_and_throw_if_fail(len(attention_weight_vector_dims) == len(attention_output_dims), "attention_weight_vector_dims")
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims=attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding= initial_embedding        
        self.use_sequence_to_vector_encoder = use_sequence_to_vector_encoder
        super(HierarchicalAttention, self).__init__(**kwargs)

    def build(self, input_shapes):
        '''
        input_shapes[0]: batch_size* snapshots* documents * sections* sentences * words
        input_shapes[1]: batch_size*snapshots* documents * sections* sentences * words* word_input_feature
        ...
        input_shapes[5]:batch_size*snapshots*snapshot_input_feature
        '''
        input_shape=input_shapes[0]
        self.embedding = Embedding(self.embedding_rows, self.embedding_dim, weights = [self.initial_embedding])
        self.attention_layers=[]
        self.encoder_layers=[]
        total_dim = len(input_shape)
        #low level to high level
        for cur_dim in xrange(total_dim - 1 , 1, -1):    
            cur_output_dim = self.attention_output_dims[cur_dim - 2]
            attention_weight_vector_dim = self.attention_weight_vector_dims[cur_dim - 2]
            attetion_layer, encoder_layer = self.create_attention_layer(attention_weight_vector_dim, cur_output_dim)
            self.attention_layers.append(attetion_layer)
            self.encoder_layers.append(encoder_layer)
    
    def create_attention_layer(self, attention_weight_vector_dim, cur_output_dim):
        if self.use_sequence_to_vector_encoder:
            return Attention(attention_weight_vector_dim), SequenceToVectorEncoder(cur_output_dim)
        else:
            return Attention(attention_weight_vector_dim), SequenceToSequenceEncoder(cur_output_dim)

    def call_attention_layer(self, input_sequence, attention_layer, encoder_layer):
        if self.use_sequence_to_vector_encoder:
                transformed_vector = encoder_layer(input_sequence)
                attention_vector = attention_layer(input_sequence)
                return K.concatenate([attention_vector, transformed_vector], axis=-1)
        else:
            return attention_layer(encoder_layer(input_sequence))
    
    def get_output_dim(self,input_shapes):
        if self.use_sequence_to_vector_encoder:
            #input_feature_dim  attention_outpout_dim + next layer output_dim
            output_dim = sum(self.attention_output_dims)            
            for input_shape in input_shapes[1:]:
                output_dim += input_shape[-1]
            output_dim += self.embedding_dim
            return output_dim 
        else:
            return input_shapes[-1][-1] + self.attention_output_dims[-1]*2
        
    @staticmethod
    def build_inputs(input_shape, input_feature_dims):
        '''
        input_shape: input shape, e.g., snapshots* documents * sections* sentences * words
        input_feature_dims: input feature dims, first one being the dim of top level feature, and last being the dim of the most low level feature 
        return inputs, first one being the most fine-grained/low level input, and last being the most coarse/high level input
        '''
        inputs = []
        check_and_throw_if_fail(len(input_shape) >= 2 , "input_shape")
        check_and_throw_if_fail(len(input_feature_dims) == len(input_shape) , "input_feature_dims")
        total_dim = len(input_shape)
        #The shape parameter of an Input does not include the first batch_size dimension
        inputs.append(Input(shape = input_shape,dtype="int32"))
        # for each level, create an input
        for cur_dim in xrange(total_dim - 1 , -1, -1):
            inputs.append(Input(shape = input_shape[:cur_dim + 1] + (input_feature_dims[cur_dim],)))        
        return inputs
   
    def call(self, inputs, mask = None):
        '''
        inputs: a list of inputs; the first layer is lowest level sequence, second layer lowest level input features, ..., the top level input features
        returns a tensor of shape: batch_size*snapshots*output_dim
        '''
        check_and_throw_if_fail(type(inputs) is  list and len(inputs) == 2 + len(self.attention_layers) , "inputs")        
        output  = self.embedding(inputs[0])
        i=1
        for attention_layer, encoder_layer  in zip(self.attention_layers, self.encoder_layers):
            output = K.concatenate([output, inputs[i]], axis=-1)
            cur_output_shape=K.int_shape(output)
            output = K.reshape(output, shape=(-1, cur_output_shape[-2], cur_output_shape[-1]))
            output = self.call_attention_layer(output,attention_layer,encoder_layer)
            output = K.reshape(output, shape=(-1,) +  cur_output_shape[1:-2] + (K.int_shape(output)[1],))
            i+=1
    
        # output: batch_size*time_steps*cacdi_snapshot_attention
        output = K.concatenate ([output, inputs[-1]],axis=-1)
        return output

    def get_output_shape_for(self, input_shapes):
        '''
        input_shapes[0]: batch_size* snapshots* documents * sections* sentences * words
        input_shapes[1]: batch_size*snapshots* documents * sections* sentences * words* word_input_feature
        ...
        input_shapes[5]:batch_size*snapshots*snapshot_input_feature
        '''
        input_shape=input_shapes[0]
        check_and_throw_if_fail(len(input_shape) >= 3, "input_shape")
        return input_shape[:2] + (self.get_output_dim(input_shapes), )

