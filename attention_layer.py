'''
Created on Jul 5, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU, LSTM, time_distributed_dense
from keras.layers.convolutional import Convolution1D
from keras.layers import Input, BatchNormalization, merge
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.engine import  InputSpec
import numpy as np

import logging

logger = logging.getLogger(__name__)

def shape(x):
    if hasattr(x, '_keras_shape'):
        return  x._keras_shape
    else:
        raise Exception('You tried to shape on a non-keras tensor "' + x.name + '". This tensor has no information  about its expected input shape.')

if K._BACKEND == 'theano':
    from theano import tensor as T
    def reverse(x):
        return x[::-1]

    def _reshape(x, shape, ndim = None):
        return T.reshape(x, shape, ndim)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def reverse(x):
        x_list = tf.unpack(x)
        x_list.reverse()
        return K.pack(x_list)

    def _reshape(x, shape, ndim = None):
        return tf.reshape(x, shape)

def check_and_throw_if_fail(condition, msg):
    '''
    condition: boolean; if condition is False, log and throw an exception
    msg: the message to log and exception if condition is False
    '''
    if not condition:
        logger.error(msg)
        raise Exception(msg)

def build_bi_directional_layer(left_to_right, right_to_left):
    '''
    Helper function that performs reshape on a tensor
    '''
    class BiDirectionalLayer(Layer):
        def call(self, inputs, mask = None):
            left_to_right = inputs[0]
            right_to_left = inputs[1]
            ndim = K.ndim(right_to_left)
            axes = [1, 0] + list(range(2, ndim))
            right_to_left = K.permute_dimensions(right_to_left, axes)
            right_to_left = reverse(right_to_left)
            right_to_left = K.permute_dimensions(right_to_left, axes)
            return K.concatenate([left_to_right, right_to_left], axis = -1)
        def get_output_shape_for(self, input_shapes):
            return input_shapes[0][:-1] + (input_shapes[0][-1] + input_shapes[1][-1],)

    check_and_throw_if_fail(K.ndim(left_to_right) >= 3 , "left_to_right")
    check_and_throw_if_fail(K.ndim(right_to_left) == K.ndim(left_to_right) , "right_to_left")
    return BiDirectionalLayer()([left_to_right, right_to_left])

class ManyToOnePooling(Layer):
    def __init__(self, mode, axis = 1, ** kwargs):
        self.mode = mode  # mode = K.max or K.mean
        self.axis = axis
        super(ReshapeLayer, self).__init__(**kwargs)
    def call(self, x, mask = None):
        return self.mode(x, axis = self.axis)
    def get_output_shape_for(self, input_shape):
        axis = self.axis % len (input_shape)
        output_shape = list(input_shape)
        del output_shape[axis]
        return output_shape
    
def reshape(x, target_shape, target_tensor_shape = None):
    '''
    Helper function that performs reshape on a tensor
    '''
    class ReshapeLayer(Layer):
        '''
        Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
        '''
        def __init__(self, target_shape, target_tensor_shape = None, ** kwargs):
            self.target_shape = tuple(target_shape)
            self.target_tensor_shape = target_tensor_shape
            super(ReshapeLayer, self).__init__(**kwargs)

        def call(self, x, mask = None):
            if self.target_tensor_shape:
                return _reshape(x, self.target_tensor_shape, ndim = len(self.target_shape))  # required by theano
            else:
                return _reshape(x, self.target_shape)

        def get_output_shape_for(self, input_shape):
            return self.target_shape

    return ReshapeLayer(target_shape = target_shape, target_tensor_shape = target_tensor_shape)(x)

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
        ui = K.tanh(time_distributed_dense(x, self.Ws, self.bs))  # batch_size, time_steps, attention_weight_vector_dim
        ai = K.exp(time_distributed_dense(ui, K.expand_dims(self.us, 1), output_dim = 1))  # batch_size, time_steps, 1
        sum_of_ai = K.sum(ai, 1, keepdims = True)  # batch_size 1 1
        ai = ai / sum_of_ai  # batch_size * time_steps * 1
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
    def __init__(self, output_dim, is_bi_directional = True, use_gru = True, **kwargs):
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        self.output_dim = output_dim
        self.is_bi_directional = is_bi_directional
        self.use_gru = use_gru
        super(SequenceToSequenceEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        if self.use_gru:
            self.encoder_left_to_right = GRU(self.output_dim, return_sequences = True)
        else:
            self.encoder_left_to_right = LSTM (self.output_dim, return_sequences = True)

        if self.is_bi_directional:
            if self.use_gru:
                self.encoder_right_to_left = GRU(self.output_dim, return_sequences = True, go_backwards = True)
            else:
                self.encoder_right_to_left = LSTM(self.output_dim, return_sequences = True, go_backwards = True)

    def call(self, x, mask = None):
        '''
        x: batch_size * time_steps* input_dim
        returns a tensor of shape batch_size * time_steps * 2*input_dim (or input_dim if not bidirectional)  
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        h1 = self. encoder_left_to_right(x)
        if self.is_bi_directional:
            h2 = self.encoder_right_to_left(x)
            return  build_bi_directional_layer(h1, h2)
        else:
            return h1

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        if self.is_bi_directional:
            return  input_shape[:-1] + (2 * self.output_dim,)
        else:
            return  input_shape[:-1] + (self.output_dim,)

class SequenceToVectorEncoder(Layer):
    '''
    Represents an encoder that transforms a sequence into a vector 
    '''
    def __init__(self, output_dim, window_size = 3 , **kwargs):
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        check_and_throw_if_fail(window_size > 0 , "window_size")
        self.output_dim = output_dim
        self.window_size = window_size
        super(SequenceToVectorEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        self.conv = Convolution1D(self.output_dim, filter_length = self.window_size, border_mode = 'same')
        self.pooling = ManyToOnePooling(mode = K.max)

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
        return (input_shape[0], self.output_dim)

class HierarchicalAttention(Layer):
    '''
    Represents a hierarchical attention layer
    One example: snapshots* documents * sections* sentences * words
    '''
    def __init__(self, top_level_input_feature_dim, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding = None, use_sequence_to_vector_encoder = False,
                 use_cnn_as_sequence_to_sequence_encoder = False, input_window_sizes = None, use_max_pooling_as_attention = False, **kwargs):
        '''
        top_feature_dim: dim of the top feature, e.g., the snapshot level feature
        attention_output_dims: attention output dimensions on different levels: e.g., section, document, sentence, word
        attention_weight_vector_dims: weight vector dimensions inside each attention layer, e.g., section, document, sentence, word
        use_sequence_to_vector_encoder: True if use sequence to vector encoder otherwise sequence to sequence encoder inside all attention layers
        '''
        check_and_throw_if_fail(len(attention_output_dims) > 0 , "attention_output_dims")
        check_and_throw_if_fail(use_max_pooling_as_attention == True or len(attention_weight_vector_dims) == len(attention_output_dims), "attention_weight_vector_dims")
        check_and_throw_if_fail(top_level_input_feature_dim >= 0 , "top_level_input_feature_dim")
        check_and_throw_if_fail(use_cnn_as_sequence_to_sequence_encoder == False or  input_window_sizes is not None  , "input_window_sizes")
        check_and_throw_if_fail(input_window_sizes is None or  len(input_window_sizes) == len(attention_output_dims) , "input_window_sizes")
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims = attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding = initial_embedding
        self.use_sequence_to_vector_encoder = use_sequence_to_vector_encoder
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.top_level_input_feature_dim = top_level_input_feature_dim
        self.use_max_pooling_as_attention = use_max_pooling_as_attention
        super(HierarchicalAttention, self).__init__(**kwargs)

    def build(self, input_shapes):
        '''
        input_shapes[0]: batch_size* snapshots* documents * sections* sentences * words
        input_shapes[1]: batch_size*snapshots* documents * sections* sentences * words* word_input_feature
        ...
        input_shapes[5]:batch_size*snapshots*snapshot_input_feature
        '''
        if not  type(input_shapes) is  list:
            input_shapes = [input_shapes]
        input_shape = input_shapes[0]
        self.input_spec = [InputSpec(shape = input_shape)]
        self.embedding = Embedding(self.embedding_rows, self.embedding_dim, weights = [self.initial_embedding])
        self.attention_layers = []
        self.encoder_layers = []
        total_dim = len(input_shape)
        # low level to high level
        for cur_dim in xrange(total_dim - 1 , 1, -1):
            cur_output_dim = self.attention_output_dims[cur_dim - 2]
            cur_sequence_length = input_shape[cur_dim]
            if self.use_max_pooling_as_attention:
                attention_weight_vector_dim = None
            else:
                attention_weight_vector_dim = self.attention_weight_vector_dims[cur_dim - 2]
            if self.use_cnn_as_sequence_to_sequence_encoder:
                cur_window_size = self.input_window_sizes[cur_dim - 2]
            else:
                cur_window_size = None
            attetion_layer, encoder_layer = self.create_attention_layer(attention_weight_vector_dim, cur_output_dim, cur_sequence_length, cur_window_size)
            self.attention_layers.append(attetion_layer)
            self.encoder_layers.append(encoder_layer)

    def create_attention_layer(self, attention_weight_vector_dim, cur_output_dim, cur_sequence_length, cur_window_size):
        if self.use_max_pooling_as_attention:
            attention = ManyToOnePooling(mode = K.max)
        else:
            attention = Attention(attention_weight_vector_dim)
        if self.use_sequence_to_vector_encoder:
            return attention, SequenceToVectorEncoder(cur_output_dim)
        else:
            if self.use_cnn_as_sequence_to_sequence_encoder:
                return attention, Convolution1D(cur_output_dim, filter_length = cur_window_size, border_mode = 'same')
            else:
                return attention, SequenceToSequenceEncoder(cur_output_dim)

    def call_attention_layer(self, input_sequence, attention_layer, encoder_layer):
        if self.use_sequence_to_vector_encoder:
                transformed_vector = encoder_layer(input_sequence)
                attention_vector = attention_layer(input_sequence)
                return merge(inputs = [attention_vector, transformed_vector], mode = 'concat')
        else:
            return attention_layer(encoder_layer(input_sequence))

    def get_attention_output_dim(self, input_shape, encoder_layer, attention_layer):
        if self.use_sequence_to_vector_encoder:
            output_shape_1 = encoder_layer.get_output_shape_for(input_shape)
            output_shape_2 = attention_layer.get_output_shape_for(input_shape)
            return output_shape_1[:-1] + (output_shape_1[-1] + output_shape_2[-1],)
        else:
            output_shape_1 = encoder_layer.get_output_shape_for(input_shape)
            output_shape_2 = attention_layer.get_output_shape_for(output_shape_1)
            return output_shape_2

    def get_output_dim(self, input_shapes):
        if self.use_sequence_to_vector_encoder:
            # input_feature_dim  attention_outpout_dim + next layer output_dim
            output_dim = sum(self.attention_output_dims)
            for input_shape in input_shapes[1:]:
                output_dim += input_shape[-1]
            output_dim += self.embedding_dim
            return output_dim
        else:
            # last input is the top level input feature; the first in the attention output dimension is for the top level
            if self.use_cnn_as_sequence_to_sequence_encoder:
                return self.top_level_input_feature_dim + self.attention_output_dims[0]
            else:
                return self.top_level_input_feature_dim + self.attention_output_dims[0] * 2

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
        total_level = len(input_shape)
        # The shape parameter of an Input does not include the first batch_size dimension
        inputs.append(Input(shape = input_shape, dtype = "int32"))
        # for each level, create an input
        for cur_level in xrange(total_level - 1 , -1, -1):
            if input_feature_dims[cur_level] > 0:
                tensor_input = Input(shape = input_shape[:cur_level + 1] + (input_feature_dims[cur_level],))
                tensor_input._level = cur_level
                tensor_input._input_dim = input_feature_dims[cur_level]
                inputs.append(tensor_input)
        return inputs

    def call(self, inputs, mask = None):
        '''
        inputs: a list of inputs; the first layer is lowest level sequence, second layer lowest level input features, ..., the top level input features
        returns a tensor of shape: batch_size*snapshots*output_dim
        '''
        if not  type(inputs) is  list:
            inputs = [inputs]
        check_and_throw_if_fail(len(inputs) <= 2 + len(self.attention_layers) , "inputs")
        output = self.embedding(inputs[0])
        cur_output_shape = list(self.input_spec[0].shape + (self.embedding_dim,))
        output = reshape(output, target_shape = cur_output_shape, target_tensor_shape = tuple(inputs[0].shape) + (self.embedding_dim,))
        level_to_input = {}
        if len(inputs) > 1:
            for tensor_input in inputs[1:]:
                check_and_throw_if_fail(hasattr(tensor_input, '_level'), "an input must have _level property")
                level_to_input[tensor_input._level] = tensor_input
        cur_level = len (self.attention_layers)
        for attention_layer, encoder_layer  in zip(self.attention_layers, self.encoder_layers):
            if cur_level in level_to_input:
                output = merge(inputs = [output, level_to_input[cur_level]], mode = 'concat')
                cur_output_shape[-1] += level_to_input[cur_level]._input_dim
            cur_output_tensor_shape = output.shape
            attention_input_shape = (-1, cur_output_shape[-2], cur_output_shape[-1])
            output = reshape(output, target_shape = attention_input_shape, target_tensor_shape = (-1, cur_output_tensor_shape[-2], cur_output_tensor_shape[-1]))
            output = self.call_attention_layer(output, attention_layer, encoder_layer)
            cur_output_shape = cur_output_shape[:-2] + [self.get_attention_output_dim(attention_input_shape, encoder_layer = encoder_layer, attention_layer = attention_layer)[-1]]
            cur_output_tensor_shape = tuple(cur_output_tensor_shape[:-2]) + (output.shape[-1],)
            output = reshape(output, target_shape = cur_output_shape, target_tensor_shape = cur_output_tensor_shape)
            cur_level -= 1
        # output: batch_size*time_steps*cacdi_snapshot_attention
        if cur_level in level_to_input:
            output = merge(inputs = [output, level_to_input[cur_level]], mode = 'concat')
        return output

    def get_output_shape_for(self, input_shapes):
        '''
        input_shapes[0]: batch_size* snapshots* documents * sections* sentences * words
        input_shapes[1]: batch_size*snapshots* documents * sections* sentences * words* word_input_feature
        ...
        input_shapes[5]:batch_size*snapshots*snapshot_input_feature
        '''
        if not  type(input_shapes) is  list:
            input_shapes = [input_shapes]
        input_shape = input_shapes[0]
        check_and_throw_if_fail(len(input_shape) >= 3, "input_shape")
        return input_shape[:2] + (self.get_output_dim(input_shapes),)

class MLPClassifierLayer(Layer):
    '''
    Represents a mlp classifier, which consists of several hidden layers followed by a softmax output layer
    '''
    def __init__(self, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function = 'softmax', **kwargs):
        '''
        input_sequence: input sequence, batch_size * time_steps * input_dim
        hidden_unit_numbers: number of hidden units of each hidden layer
        hidden_unit_activation_functions: activation function of hidden layers
        output_dim: output dimension
        returns a tensor of shape: batch_size*time_steps*output_dim
        '''
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        check_and_throw_if_fail(len(hidden_unit_numbers) == len(hidden_unit_activation_functions) , "hidden_unit_numbers")
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self.output_activation_function = output_activation_function
        if hidden_unit_numbers:
            self.uses_learning_phase = True
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = []
        ndim = len(input_shape)
        for hidden_unit_number, hidden_unit_activation_function in zip(self.hidden_unit_numbers, self.hidden_unit_activation_functions):
            dense = Dense(hidden_unit_number, activation = hidden_unit_activation_function)
            if ndim == 3:
                dense = TimeDistributed(dense)
            norm = BatchNormalization()
            self.layers.append(dense)
            self.layers.append(norm)

        dense = Dense(self.output_dim, activation = self.output_activation_function)
        if ndim == 3:
            dense = TimeDistributed(dense)
        self.layers.append(dense)

    def call(self, x, mask = None):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

class ClassifierWithHierarchicalAttention(Layer):
    def __init__(self, top_level_input_feature_dim, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim,
                 initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function = 'softmax',
                 use_cnn_as_sequence_to_sequence_encoder = False, input_window_sizes = None, use_max_pooling_as_attention = False, **kwargs):
        self.top_level_input_feature_dim = top_level_input_feature_dim
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims = attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding = initial_embedding
        self.use_sequence_to_vector_encoder = use_sequence_to_vector_encoder
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self. output_activation_function = output_activation_function
        if hidden_unit_numbers:
            self.uses_learning_phase = True
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.use_max_pooling_as_attention = use_max_pooling_as_attention
        super(ClassifierWithHierarchicalAttention, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.hierarchical_attention = HierarchicalAttention(self.top_level_input_feature_dim, self.attention_output_dims, self.attention_weight_vector_dims,
                                                              self.embedding_rows, self.embedding_dim, self.initial_embedding, self.use_sequence_to_vector_encoder,
                                                              self.use_cnn_as_sequence_to_sequence_encoder , self.input_window_sizes , self.use_max_pooling_as_attention)
        self.mlp_softmax_classifier = MLPClassifierLayer(self.output_dim, self.hidden_unit_numbers, self.hidden_unit_activation_functions, self. output_activation_function)

    def call(self, inputs, mask = None):
        output = self.hierarchical_attention(inputs)
        output = self.mlp_softmax_classifier(output)
        return output

    def get_output_shape_for(self, input_shapes):
        return self.hierarchical_attention.get_output_shape_for(input_shapes)[:-1] + (self.output_dim,)

