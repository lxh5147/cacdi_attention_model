'''
Created on Jul 5, 2016

@author: lxh5147
'''
import logging

from keras import backend as K
from keras.layers import Lambda, Embedding,TimeDistributed,Dense,Dropout,Layer, Convolution1D, GRU, LSTM, time_distributed_dense,Input, BatchNormalization, merge
from keras.engine import InputSpec
from keras import initializations
from keras.utils.generic_utils import get_from_module
from keras import regularizers

logger = logging.getLogger()

def get_mask(input, padding_id=1):
    return K.cast( K.not_equal (input, padding_id), dtype='int32')


def add_mask(input, mask):
    if mask is None:
        return input
    else:
        return AddMaskToSequence()([input,mask])


class AddMaskToSequence(Layer):
    '''
    Attach a mask to an input sequence
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AddMaskToSequence, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return inputs[-1]

    def call(self, inputs, mask=None):
        return inputs[0]

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

def mask_sequence(input, mask_value, is_vector_sequence=False):
    return MaskSequence(mask_value, is_vector_sequence=is_vector_sequence)(input)


class MaskSequence(Layer):
    '''
    Attach a mask to an input sequence
    '''
    def __init__(self, mask_value=0., is_vector_sequence=False,**kwargs):
        self.supports_masking = True
        self.mask_value = mask_value
        self.is_vector_sequence = is_vector_sequence
        super(MaskSequence, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        if self.is_vector_sequence:
            return Lambda(lambda x: K.cast( K.any(K.not_equal(x, self.mask_value), axis=-1), dtype='int32'),
                          output_shape= lambda input_shape: input_shape[:-1])(input)
        else:
            return Lambda( lambda x: get_mask( x, self.mask_value)) (input)

    def call(self, input, mask=None):
        # to create a new tensor
        output =  input + 0
        # apply mask
        output,_ = apply_mask(output, mask)
        # copy properties required by attention model
        if hasattr(input, "_level"):
            output._level = getattr(input,"_level")
        if hasattr(input, "_input_dim"):
            output._input_dim = getattr(input, "_input_dim")
        return output

    def get_config(self):
        config = {'mask_value': self.mask_value,
                  'is_vector_sequence':self.is_vector_sequence}
        base_config = super(MaskSequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def remove_mask(input, apply_mask_before_remove=True):
    return RemoveMask(apply_mask_before_remove=apply_mask_before_remove)(input)


def apply_mask(input_tensor, input_mask):
    assert K.ndim(input_tensor)>=2
    assert input_mask is None or K.ndim(input_mask) == K.ndim(input_tensor) or K.ndim(input_mask) == K.ndim(input_tensor) -1
    if input_mask is None:
        return input_tensor, input_mask
    else:
        # input_tensor: nb_samples,..., time_steps, input_dim
        # input_mask: nb_samples, ..., time_steps
        if K.ndim(input_mask) == K.ndim(input_tensor) - 1:
            mask = K.expand_dims(input_mask)
        else:
            mask = input_mask
        mask = K.cast(mask, K.dtype(input_tensor))
        return input_tensor * mask, mask


class RemoveMask(Layer):
    '''
    Attach a mask to an input sequence, and then remove the mask
    '''
    def __init__(self, apply_mask_before_remove=True, **kwargs):
        self.supports_masking = True
        self.apply_mask_before_remove = apply_mask_before_remove
        super(RemoveMask, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if self.apply_mask_before_remove:
            output,_=  apply_mask(x,mask)
        else:
            output = x
        # return a new tensor
        return output + 0

def merge_mask(mask_one, mask_two):
    if mask_one is None:
        return mask_two
    if mask_two is None:
        return mask_one

    assert K.ndim(mask_one) == K.ndim(mask_two)
    assert K.dtype(mask_one) == K.dtype(mask_two)

    new_mask = mask_one + mask_two
    return K.cast(K.not_equal(new_mask, 0), dtype=K.dtype(mask_one))


def get_mask_of_up_level(mask):
    # e.g., from word level mask to sentence level mask: nb_samples, sentences, words -> # nb_samples, sentences
    if mask is None:
        return mask
    assert K.ndim(mask) >= 3
    new_mask = K.sum(mask,axis=-1)
    return K.cast(K.not_equal(new_mask, 0), dtype=K.dtype(mask))

def get_copy_of(dict):
    if dict is None:
        return None
    else:
        return dict.copy()

class ComposedLayer(Layer):
    '''A layer that contains a set of children layers to complete its logic
    '''
    def __init__(self, **kwargs):
        # trainable weights for this layer, not including trainable weights of any children layer
        super(ComposedLayer, self).__init__(**kwargs)
        self._trainable_weights = []
        self._updates = []
        # non-trainable weights for this layer, not including non-trainable weights of any children layer
        self._non_trainable_weights = []
        self._stateful = False
        self._uses_learning_phase = False
        self._constraints = {}
        self._regularizers = []
        self._layers = []

    @property
    def updates(self):
        updates = []
        updates += self._updates
        for layer in self._layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates

    @property
    def constraints(self):
        cons = {}
        for key,value in self._constraints.items():
            cons[key] = value
        for layer in self._layers:
            for key, value in layer.constraints.items():
                if key in cons:
                    raise Exception('Received multiple constraints '
                                    'for one weight tensor: ' + str(key))
                cons[key] = value
        return cons

    @constraints.setter
    def constraints(self, _):
        pass

    @property
    def regularizers(self):
        regs = []
        regs += self._regularizers
        for layer in self._layers:
            regs += layer.regularizers
        return regs

    @regularizers.setter
    def regularizers(self, _):
        pass

    @property
    def stateful(self):
        if self._stateful:
            return self._stateful
        return any([(hasattr(layer, 'stateful') and layer.stateful) for layer in self._layers])

    def reset_states(self):
        for layer in self._layers:
            if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
                layer.reset_states()

    @property
    def uses_learning_phase(self):
        '''True if any layer in the graph uses it.
        '''
        if self._uses_learning_phase:
            return self._uses_learning_phase
        layers_learning_phase = any([layer.uses_learning_phase for layer in self._layers])
        regs_learning_phase = any([reg.uses_learning_phase for reg in self.regularizers])
        return layers_learning_phase or regs_learning_phase

    @uses_learning_phase.setter
    def uses_learning_phase(self, _):
        pass

    @property
    def trainable_weights(self):
        weights = []
        weights += self._trainable_weights
        for layer in self._layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        weights += self._non_trainable_weights
        for layer in self._layers:
            weights += layer.non_trainable_weights
        return weights

    @trainable_weights.setter
    def trainable_weights(self, _):
        pass

    @non_trainable_weights.setter
    def non_trainable_weights(self, _):
        pass


def shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise Exception(
            'You tried to shape on a non-keras tensor "' + x.name + '". This tensor has no information  about its expected input shape.')


if K._BACKEND == 'theano':
    from theano import tensor as T


    def reverse(x):
        return x[::-1]


    def _reshape(x, shape, ndim=None):
        return T.reshape(x, shape, ndim)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf


    def reverse(x):
        x_list = tf.unpack(x)
        x_list.reverse()
        return K.pack(x_list)


    def _reshape(x, shape, ndim=None):
        return tf.reshape(x, shape)


def check_and_throw_if_fail(condition, msg):
    '''
    condition: boolean; if condition is False, log and throw an exception
    msg: the message to log and exception if condition is False
    '''
    if not condition:
        raise Exception(msg)


def build_bi_directional_layer(left_to_right, right_to_left):
    '''
    Helper function that performs reshape on a tensor
    '''

    class BiDirectionalLayer(Layer):

        def __init__(self, **kwargs):
            self.supports_masking = True
            super(BiDirectionalLayer, self).__init__(**kwargs)

        def call(self, inputs, mask=None):
            left_to_right = inputs[0]
            right_to_left = inputs[1]
            ndim = K.ndim(right_to_left)
            axes = [1, 0] + list(range(2, ndim))
            right_to_left = K.permute_dimensions(right_to_left, axes)
            right_to_left = reverse(right_to_left)
            right_to_left = K.permute_dimensions(right_to_left, axes)
            return K.concatenate([left_to_right, right_to_left], axis=-1)

        def compute_mask(self, x, mask):
            if mask is None:
                return None
            return mask[0]

        def get_output_shape_for(self, input_shapes):
            return input_shapes[0][:-1] + (input_shapes[0][-1] + input_shapes[1][-1],)

    check_and_throw_if_fail(K.ndim(left_to_right) >= 3, "left_to_right")
    check_and_throw_if_fail(K.ndim(right_to_left) == K.ndim(left_to_right), "right_to_left")
    return BiDirectionalLayer()([left_to_right, right_to_left])

min = K.min
max = K.max
sum = K.sum
avg = K.mean


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'many_to_one_mode', instantiate=False, kwargs=kwargs)


class ManyToOnePooling(Layer):

    MAX = 10000000000.0

    def __init__(self, mode, axis=1, **kwargs):
        self.mode = get(mode)  # mode = max, sum or mean
        self.axis = axis
        self.supports_masking = True
        super(ManyToOnePooling, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is None:
            return self.mode(x, axis=self.axis)
        else:
            # x: nb_samples, time_steps, input_dim
            # mask: nb_samples, time_steps
            # mask and x must have the same dim
            x,mask = apply_mask(x,mask)
            # if mode == K.sum:
            if self.mode == K.sum:
                return self.mode(x,axis=self.axis)
            elif self.mode == K.min:
                x += (1-mask) * ManyToOnePooling.MAX
                return self.mode(x,axis=self.axis)
            elif self.mode == K.max:
                x += (1-mask) * (0-ManyToOnePooling.MAX)
                return self.mode(x,axis=self.axis)
            elif self.mode == K.mean:
                sum = K.sum(x,axis=self.axis) # nb_samples, input_dim
                len = K.sum (mask,self.axis) #nb_samples, 1
                # in case len is zero
                len += K.cast(K.equal(len, 0), dtype=K.dtype(len))
                # if len of a sample is zero, its sum should also be zero
                # so it is fine to change len to 1
                return sum / len
            else:
                return self.mode(x, axis=self.axis)

    def compute_mask(self, x, mask):
        return None

    def get_output_shape_for(self, input_shape):
        axis = self.axis % len(input_shape)
        output_shape = list(input_shape)
        del output_shape[axis]
        return tuple(output_shape)


def reshape(x, target_shape, target_tensor_shape=None):
    '''
    Helper function that performs reshape on a tensor
    '''

    class ReshapeLayer(Layer):
        '''
        Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
        '''

        def __init__(self, target_shape, target_tensor_shape=None, **kwargs):
            self.target_shape = tuple(target_shape)
            self.target_tensor_shape = target_tensor_shape
            super(ReshapeLayer, self).__init__(**kwargs)

        def call(self, x, mask=None):
            if self.target_tensor_shape:
                return _reshape(x, self.target_tensor_shape, ndim=len(self.target_shape))  # required by theano
            else:
                return _reshape(x, self.target_shape)

        def get_output_shape_for(self, input_shape):
            output_shape = []
            for shape in self.target_shape:
                if isinstance(shape, int) and shape != -1:
                    output_shape.append(shape)
                else:
                    output_shape.append(None)

            return tuple(output_shape)

    return ReshapeLayer(target_shape=target_shape, target_tensor_shape=target_tensor_shape)(x)


# https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
# Seems to use a global context vector, which doesn't make sense for a
# *context* vector.
# We can improvement in two ways, which we can also combine.
#
# 1. Context embedding
# u_s = W_s[s_i], use a different context vector for each sentences. They are
# learned jointly with models parameters. Possibly better to pretrain those
# embeddings, with something like language model learning jointly word and
# context embedding.
#
# 2. Features dependent representation
# u_s = MLP(W_c Z + b)
# u_s is calculated based on the features of the current hierarchical level Z.
# For example, for a given sentence, we have tensor of features Z, we compute
# u_s based on this and the attention weight is computed with u_i and u_s, the
# token reprensation and the context representation respectively.
#
# * Combine 1. and 2.
# We can do something like u_s = MLP(W_c Z + V_c W_s[s_i] + b)

# Default GlobalAttention
# 1. EmbeddingBasedAttention
# 2. FeatureBasedAttention
# *  ContextualAttention


class Attention(Layer):
    '''
    Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
    '''

    def __init__(self, attention_weight_vector_dim, init_Ws='glorot_uniform',
                 init_us='uniform', attention_input_dropout=0.,
                 attention_dropout=0., weight_regularizer=None, **kwargs):
        '''
        attention_weight_vector_dim: dimension of the attention weight vector
        element_wise_output_transformer: element-wise output transformer,e.g., K.sigmoid
        '''
        check_and_throw_if_fail(attention_weight_vector_dim > 0, "attention_weight_vector_dim")
        self.init_Ws = initializations.get(init_Ws)
        self.init_us = initializations.get(init_us)
        self.attention_weight_vector_dim = attention_weight_vector_dim
        self.supports_masking = True
        self.attention_input_dropout = attention_input_dropout
        self.attention_dropout = attention_dropout

        if self.attention_input_dropout or self.attention_dropout:
            self.uses_learning_phase = True

        self.weight_regularizer = weight_regularizer
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: batch_size * time_steps* input_dim
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        input_dim = input_shape[2]
        self.Ws = self.init_Ws((input_dim, self.attention_weight_vector_dim))
        self.bs = K.zeros((self.attention_weight_vector_dim,))
        self.us = self.init_us((self.attention_weight_vector_dim,))
        self.trainable_weights = [self.Ws, self.bs, self.us]

        self.regularizers = []

        if self.weight_regularizer:
            self.Ws_regularizer = regularizers.get(get_copy_of(self.weight_regularizer))
            self.Ws_regularizer.set_param(self.Ws)
            self.regularizers.append(self.Ws_regularizer)

            self.bs_regularizer = regularizers.get(get_copy_of(self.weight_regularizer))
            self.bs_regularizer.set_param(self.bs)
            self.regularizers.append(self.bs_regularizer)

            self.us_regularizer = regularizers.get(get_copy_of(self.weight_regularizer))
            self.us_regularizer.set_param(self.us)
            self.regularizers.append(self.us_regularizer)


    def call(self, x, mask=None):
        '''
        x: batch_size * time_steps* input_dim
        '''

        # dropout for x
        x = K.dropout(x=x, level=self.attention_input_dropout)

        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        ui = K.tanh(time_distributed_dense(x, self.Ws, self.bs))  # batch_size, time_steps, attention_weight_vector_dim

        # dropout for ui
        ui = K.dropout(x=ui, level=self.attention_dropout)

        ai = K.exp(time_distributed_dense(ui, K.expand_dims(self.us, 1), output_dim=1))  # batch_size, time_steps, 1
        ai, _ = apply_mask(ai,mask)
        sum_of_ai = K.sum(ai, 1, keepdims=True)  # batch_size 1 1
        # in case sum_of_ai is zero
        sum_of_ai += K.cast( K.equal(sum_of_ai, 0), dtype=K.dtype(ai))
        ai = ai / sum_of_ai  # batch_size * time_steps * 1
        # batch_size *time_steps * input_dim -> batch_size* input_dim
        output = K.sum(ai * x, 1)
        return output

    def compute_mask(self, x, mask):
        return None

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = {'attention_weight_vector_dim': self.attention_weight_vector_dim,
                  'init_Ws': self.init_Ws.__name__,
                  'init_us': self.init_us.__name__,
                  'weight_regularizer': self.weight_regularizer,
                  }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceToSequenceEncoder(ComposedLayer):
    '''
    Represents an encoder that transforms input sequence to another sequence
    '''

    def __init__(self, output_dim, is_bi_directional=True, use_gru=True,
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 **kwargs):
        check_and_throw_if_fail(output_dim > 0, "output_dim")
        super(SequenceToSequenceEncoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.is_bi_directional = is_bi_directional
        self.use_gru = use_gru
        self.supports_masking = True
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self._build_layers()

    def _build_layers(self):
        if self.use_gru:
            self.encoder_left_to_right = GRU(self.output_dim,
                                             return_sequences=True,
                                             W_regularizer=get_copy_of(self.W_regularizer),
                                             U_regularizer=get_copy_of(self.U_regularizer),
                                             b_regularizer=get_copy_of(self.b_regularizer),
                                             dropout_W=self.dropout_W,
                                             dropout_U=self.dropout_U,
                                             consume_less='gpu')
        else:
            self.encoder_left_to_right = LSTM(self.output_dim,
                                              return_sequences=True,
                                              W_regularizer=get_copy_of(self.W_regularizer),
                                              U_regularizer=get_copy_of(self.U_regularizer),
                                              b_regularizer=get_copy_of(self.b_regularizer),
                                              dropout_W=self.dropout_W,
                                              dropout_U=self.dropout_U,
                                              consume_less='gpu')
        self._layers.append(self.encoder_left_to_right)

        if self.is_bi_directional:
            if self.use_gru:
                self.encoder_right_to_left = GRU(self.output_dim,
                                                 return_sequences=True,
                                                 go_backwards=True,
                                                 W_regularizer=get_copy_of(self.W_regularizer),
                                                 U_regularizer=get_copy_of(self.U_regularizer),
                                                 b_regularizer=get_copy_of(self.b_regularizer),
                                                 dropout_W=self.dropout_W,
                                                 dropout_U=self.dropout_U,
                                                 consume_less='gpu')
            else:
                self.encoder_right_to_left = LSTM(self.output_dim,
                                                  return_sequences=True,
                                                  go_backwards=True,
                                                  W_regularizer=get_copy_of(self.W_regularizer),
                                                  U_regularizer=get_copy_of(self.U_regularizer),
                                                  b_regularizer=get_copy_of(self.b_regularizer),
                                                  dropout_W=self.dropout_W,
                                                  dropout_U=self.dropout_U,
                                                  consume_less='gpu')

            self._layers.append(self.encoder_right_to_left)

    def call(self, x, mask=None):
        '''
        x: batch_size * time_steps* input_dim
        returns a tensor of shape batch_size * time_steps * 2*input_dim (or input_dim if not bidirectional)
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        # use the mask associated with x automatically
        h1 = self.encoder_left_to_right(x)
        if self.is_bi_directional:
            h2 = self.encoder_right_to_left(x)
            return build_bi_directional_layer(h1, h2)
        else:
            return h1

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        if self.is_bi_directional:
            return input_shape[:-1] + (2 * self.output_dim,)
        else:
            return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'is_bi_directional': self.is_bi_directionalis_bi_directional,
                  'use_gru': self.use_gru,
                  'W_regularizer': self.W_regularizer,
                  'U_regularizer': self.U_regularizer,
                  'b_regularizer': self.b_regularizer,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  }
        base_config = super(SequenceToSequenceEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceToVectorEncoder(ComposedLayer):
    '''
    Represents an encoder that transforms a sequence into a vector
    '''

    def __init__(self, output_dim, window_size=3, pooling_mode='max', **kwargs):
        check_and_throw_if_fail(output_dim > 0, "output_dim")
        check_and_throw_if_fail(window_size > 0, "window_size")
        super(SequenceToVectorEncoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.window_size = window_size
        self.supports_masking = True
        self.pooling_mode = pooling_mode
        self._build_layers()

    def _build_layers(self):
        self.conv = Convolution1D(self.output_dim, filter_length=self.window_size, border_mode='same')
        self.pooling = ManyToOnePooling(mode=self.pooling_mode)
        self._layers.append(self.conv)
        self._layers.append(self.pooling)

    def call(self, x, mask=None):
        '''
        x: batch_size * time_steps* input_dim
        Returns a tensor of the shape: batch_size * output_dim
        '''
        check_and_throw_if_fail(K.ndim(x) == 3, "x")
        # apply mask to input sequence
        x = remove_mask(x)
        output = self.conv(x)
        # apply mask to output
        output = add_mask(output, mask)
        output = self.pooling(output)  # batch_size * output_dim
        return output

    def compute_mask(self, x, mask):
        return None

    def get_output_shape_for(self, input_shape):
        '''
        input_shape: input shape
        '''
        check_and_throw_if_fail(len(input_shape) == 3, "input_shape")
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'window_size': self.window_size,
                  'pooling_mode':self.pooling_mode}
        base_config = super(SequenceToVectorEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HierarchicalAttention(ComposedLayer):
    '''
    Represents a hierarchical attention layer
    One example: snapshots* documents * sections* sentences * words
    '''

    def __init__(self, tree_shape, top_level_input_feature_dim, attention_output_dims, attention_weight_vector_dims, embedding_rows,
                 embedding_dim, initial_embedding=None, use_sequence_to_vector_encoder=False,
                 use_cnn_as_sequence_to_sequence_encoder=False, input_window_sizes=None, pooling_mode=None, norm_inner_output = False,
                 hierarchical_layer_dropout_W=0., hierarchical_layer_dropout_U=0.,
                 attention_input_dropout=0., attention_dropout=0.,
                 **kwargs):
        '''
        top_feature_dim: dim of the top feature, e.g., the snapshot level feature
        attention_output_dims: attention output dimensions on different levels: e.g., section, document, sentence, word
        attention_weight_vector_dims: weight vector dimensions inside each attention layer, e.g., section, document, sentence, word
        use_sequence_to_vector_encoder: True if use sequence to vector encoder otherwise sequence to sequence encoder inside all attention layers
        '''
        check_and_throw_if_fail(len(tree_shape) > 0, "tree_shape")
        check_and_throw_if_fail(len(attention_output_dims) > 0, "attention_output_dims")
        check_and_throw_if_fail(len(tree_shape) == len(attention_output_dims) + 1, "tree_shape")
        check_and_throw_if_fail(pooling_mode or len(attention_weight_vector_dims) == len(attention_output_dims),
                                "attention_weight_vector_dims")
        check_and_throw_if_fail(top_level_input_feature_dim >= 0, "top_level_input_feature_dim")
        check_and_throw_if_fail(use_cnn_as_sequence_to_sequence_encoder == False or input_window_sizes is not None,
                                "input_window_sizes")
        check_and_throw_if_fail(input_window_sizes is None or len(input_window_sizes) == len(attention_output_dims),
                                "input_window_sizes")
        super(HierarchicalAttention, self).__init__(**kwargs)
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims = attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding = initial_embedding
        self.use_sequence_to_vector_encoder = use_sequence_to_vector_encoder
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.top_level_input_feature_dim = top_level_input_feature_dim
        self.pooling_mode = pooling_mode
        self.tree_shape = tree_shape
        self.supports_masking = True
        self.norm_inner_output = norm_inner_output
        self.hierarchical_layer_dropout_W = hierarchical_layer_dropout_W
        self.hierarchical_layer_dropout_U = hierarchical_layer_dropout_U
        self.attention_input_dropout = attention_input_dropout
        self.attention_dropout = attention_dropout
        self._build_layers()

    def _build_layers(self):
        input_shape = self.tree_shape # not including the nb_samples
        self.embedding = Embedding(self.embedding_rows, self.embedding_dim, weights=[self.initial_embedding])
        self.norm_layers = []
        self.attention_layers = []
        self.encoder_layers = []
        total_dim = len(input_shape)
        # low level to high level
        for cur_dim in xrange(total_dim - 1, 0, -1):
            cur_output_dim = self.attention_output_dims[cur_dim - 1]
            cur_sequence_length = input_shape[cur_dim]
            if self.pooling_mode:
                attention_weight_vector_dim = None
            else:
                attention_weight_vector_dim = self.attention_weight_vector_dims[cur_dim - 1]
            if self.use_cnn_as_sequence_to_sequence_encoder:
                cur_window_size = self.input_window_sizes[cur_dim - 1]
            else:
                cur_window_size = None
            norm_layer, attention_layer, encoder_layer = self.create_attention_layer(attention_weight_vector_dim, cur_output_dim, cur_sequence_length, cur_window_size)
            self.norm_layers.append(norm_layer)
            self.attention_layers.append(attention_layer)
            self.encoder_layers.append(encoder_layer)
            # if use cnn as the encoder layer, add a drop out layer for cnn
            if self.use_cnn_as_sequence_to_sequence_encoder:
                dropout_layer = Dropout(p=self.hierarchical_layer_dropout_W)
                # add to layers of current container
                self._layers.append(dropout_layer)
                # associate dropout to the encoder
                encoder_layer.dropout = dropout_layer

        self._layers += ([self.embedding] + self.norm_layers + self.attention_layers + self.encoder_layers)

    def create_attention_layer(
            self, attention_weight_vector_dim, cur_output_dim,
            cur_sequence_length, cur_window_size):
        norm = BatchNormalization(mode=1)
        if self.pooling_mode:  # max, sum or None
            attention = ManyToOnePooling(mode=self.pooling_mode)
        else:
            attention = Attention(
                attention_weight_vector_dim,
            attention_dropout=self.attention_dropout,
            attention_input_dropout=self.attention_input_dropout)
        if self.use_sequence_to_vector_encoder:
            return norm, attention, SequenceToVectorEncoder(cur_output_dim)
        else:
            if self.use_cnn_as_sequence_to_sequence_encoder:
                logger.info('relu')
                cnn = Convolution1D(cur_output_dim, filter_length=cur_window_size,
                                  border_mode='same', activation="relu")
                return norm, attention, cnn
            else:
                seq2seq = SequenceToSequenceEncoder(
                    cur_output_dim,
                    dropout_W=self.hierarchical_layer_dropout_W,
                    dropout_U=self.hierarchical_layer_dropout_U)
                return norm, attention, seq2seq

    def call_attention_layer(self, input_sequence, norm_layer, attention_layer, encoder_layer, input_mask = None):
        if self.norm_inner_output:
            # before apply norm, apply the mask if applicable
            input_sequence = remove_mask(input_sequence)
            input_sequence = norm_layer(input_sequence)
        # mask the input sequence
        input_sequence = add_mask(input_sequence, input_mask)
        if self.use_sequence_to_vector_encoder:
            transformed_vector = encoder_layer(input_sequence)
            attention_vector = attention_layer(input_sequence)
            return merge(inputs=[attention_vector, transformed_vector], mode='concat')
        else:
            if self.use_cnn_as_sequence_to_sequence_encoder:
                # apply the dropout to the input before running cnn encoder
                assert encoder_layer.dropout
                input_sequence = encoder_layer.dropout(input_sequence)
                # since conv does not support mask, remove mask first
                input_sequence = remove_mask(input_sequence)
                output = encoder_layer(input_sequence)
                output = add_mask(output,input_mask)
                return attention_layer(output)
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
            # last input is the top level input feature;
            # the first in the attention output dimension is for the top level
            if self.use_cnn_as_sequence_to_sequence_encoder:
                return self.top_level_input_feature_dim + self.attention_output_dims[0]
            else:
                return self.top_level_input_feature_dim + self.attention_output_dims[0] * 2

    @staticmethod
    def build_inputs(tree_shape, input_feature_dims):
        '''
        input_shape: input shape, e.g., snapshots* documents * sections* sentences * words
        input_feature_dims: input feature dims, first one being the dim of top level feature, and last being the dim of the most low level feature
        return inputs, first one being the most fine-grained/low level input, and last being the most coarse/high level input
        '''
        inputs = []
        check_and_throw_if_fail(len(tree_shape) >= 2, "input_shape")
        check_and_throw_if_fail(len(input_feature_dims) == len(tree_shape), "input_feature_dims")
        total_level = len(tree_shape)
        # The shape parameter of an Input does not include the first batch_size dimension
        inputs.append(Input(shape=tree_shape, dtype="int32"))
        # for each level, create an input
        for cur_level in xrange(total_level - 1, -1, -1):
            if input_feature_dims[cur_level] > 0:
                tensor_input = Input(shape=tree_shape[:cur_level + 1] + (input_feature_dims[cur_level],))
                tensor_input._level = cur_level
                tensor_input._input_dim = input_feature_dims[cur_level]
                inputs.append(tensor_input)
        return inputs

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

    def call(self, inputs, mask=None):
        '''
        inputs: a list of inputs; the first layer is lowest level sequence, second layer lowest level input features, ..., the top level input features
        returns a tensor of shape: batch_size*snapshots*output_dim
        '''
        if type(inputs) is not list:
            inputs = [inputs]

        if mask is not None and type(mask) is not list:
            mask = [mask]

        check_and_throw_if_fail(len(inputs) <= 2 + len(self.attention_layers), "inputs")

        if mask:
            cur_mask = mask[0]
        else:
            cur_mask = None

        output = self.embedding(remove_mask(inputs[0]))

        cur_output_shape = list(self.input_spec[0].shape + (self.embedding_dim,))
        output = reshape(output, target_shape=cur_output_shape,
                         target_tensor_shape=tuple(inputs[0].shape) + (self.embedding_dim,))
        level_to_input = {}
        if len(inputs) > 1:
            for tensor_input in inputs[1:]:
                check_and_throw_if_fail(hasattr(tensor_input, '_level'), "an input must have _level property")
                level_to_input[tensor_input._level] = tensor_input
        cur_level = len(self.attention_layers)
        if mask:
            input_masks = iter(mask[1:])
        else:
            input_masks = None

        for norm_layer, attention_layer, encoder_layer  in zip(self.norm_layers,self.attention_layers, self.encoder_layers):
            if cur_level in level_to_input:
                # each input has its own mask
                if input_masks:
                    input_mask = next(input_masks)
                else:
                    input_mask = None
                cur_input = remove_mask(level_to_input[cur_level])
                output = merge(inputs=[output, cur_input], mode='concat')
                # update current mask
                if input_mask:
                    cur_mask = Lambda(lambda inputs: merge_mask(inputs[0],inputs[1]),
                                      output_shape = lambda input_shapes: input_shapes[0])([cur_mask, input_mask])

                cur_output_shape[-1] += level_to_input[cur_level]._input_dim

            cur_output_tensor_shape = output.shape
            attention_input_shape = (-1, cur_output_shape[-2], cur_output_shape[-1])
            output = reshape(output, target_shape=attention_input_shape,
                             target_tensor_shape=(-1, cur_output_tensor_shape[-2], cur_output_tensor_shape[-1]))

            if cur_mask:
                input_mask_shape = (-1, cur_output_shape[-2])
                input_mask_tensor_shape = (-1, cur_output_tensor_shape[-2])
                attention_input_mask = reshape(cur_mask,
                            target_shape=input_mask_shape,
                            target_tensor_shape=input_mask_tensor_shape)
            else:
                attention_input_mask = None

            output = self.call_attention_layer(output, norm_layer,attention_layer, encoder_layer, attention_input_mask)
            output = remove_mask(output)

            if cur_mask:
                cur_mask = Lambda(lambda input_mask: get_mask_of_up_level(input_mask),
                                  output_shape=lambda input_shape: input_shape[:-1])(cur_mask)

            cur_output_shape = cur_output_shape[:-2] + [
                self.get_attention_output_dim(attention_input_shape, encoder_layer=encoder_layer,
                                              attention_layer=attention_layer)[-1]]
            cur_output_tensor_shape = tuple(cur_output_tensor_shape[:-2]) + (output.shape[-1],)
            output = reshape(output, target_shape=cur_output_shape, target_tensor_shape=cur_output_tensor_shape)
            cur_level -= 1

        # output: batch_size*time_steps*cacdi_snapshot_attention
        if cur_level in level_to_input:
            cur_input = remove_mask(level_to_input[cur_level])
            output = merge(inputs=[output, cur_input], mode='concat')

        return output

    def compute_mask(self, inputs, mask):
        if mask is None:
            return None

        if type(inputs) is not list:
            inputs = [inputs]

        if type(mask) is not list:
            mask = [mask]

        cur_mask = mask[0]
        if cur_mask is None:
            return None

        input_masks = iter(mask[1:])

        level_to_input = [tensor_input._level for tensor_input in inputs[1:] ]

        for  cur_level in range( len(self.attention_layers),0,-1):
            if cur_level in level_to_input:
                input_mask = next(input_masks)
                cur_mask = Lambda(lambda inputs: merge_mask(inputs[0], inputs[1]),
                                  output_shape=lambda input_shapes: input_shapes[0])([cur_mask, input_mask])

            cur_mask = Lambda(lambda input_mask: get_mask_of_up_level(input_mask), output_shape=lambda input_shape: input_shape[:-1])(cur_mask)

        # cur_mask: nb_samples, time_steps
        if 0 in level_to_input:
            input_mask = next(input_masks)
            cur_mask = Lambda(lambda inputs: merge_mask(inputs[0], inputs[1]),
                              output_shape=lambda input_shapes: input_shapes[0])([cur_mask, input_mask])

        return cur_mask

    def get_output_shape_for(self, input_shapes):
        '''
        input_shapes[0]: batch_size* snapshots* documents * sections* sentences * words
        input_shapes[1]: batch_size*snapshots* documents * sections* sentences * words* word_input_feature
        ...
        input_shapes[5]:batch_size*snapshots*snapshot_input_feature
        '''
        if not type(input_shapes) is list:
            input_shapes = [input_shapes]
        input_shape = input_shapes[0]
        check_and_throw_if_fail(len(input_shape) >= 3, "input_shape")
        # returns a sequence of vectors corresponding to the 2^nd top layer
        return input_shape[:2] + (self.get_output_dim(input_shapes),)

    def get_config(self):
        config = {'tree_shape': self.tree_shape,
                  'top_level_input_feature_dim': self.top_level_input_feature_dim,
                  'attention_output_dims': self.attention_output_dims,
                  'attention_weight_vector_dims': self.attention_weight_vector_dims,
                  'embedding_rows': self.embedding_rows,
                  'embedding_dim': self.embedding_dim,
                  'initial_embedding': self.initial_embedding,
                  'use_sequence_to_vector_encoder': self.use_sequence_to_vector_encoder,
                  'use_cnn_as_sequence_to_sequence_encoder': self.use_cnn_as_sequence_to_sequence_encoder,
                  'input_window_sizes': self.input_window_sizes,
                  'pooling_mode': self.pooling_mode,
                  'norm_inner_output': self.norm_inner_output}
        base_config = super(HierarchicalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MLPClassifierLayer(ComposedLayer):
    '''
    Represents a mlp classifier, which consists of several hidden layers followed by a softmax output layer
    '''

    def __init__(self, output_dim,
                 hidden_unit_numbers,
                 hidden_unit_activation_functions,
                 output_activation_function='softmax',
                 use_sequence_input=True,
                 norm_inner_output = False,
                 weight_regularizer_batch_norm = None,
                 weight_regularizer_hidden=None,
                 weight_regularizer_mlp_output = None,
                 input_drop_out_rate=0.,
                 **kwargs):
        '''
        input_sequence: input sequence, batch_size * time_steps * input_dim
        hidden_unit_numbers: number of hidden units of each hidden layer
        hidden_unit_activation_functions: activation function of hidden layers
        output_dim: output dimension
        returns a tensor of shape: batch_size*time_steps*output_dim
        '''
        check_and_throw_if_fail(output_dim > 0, "output_dim")
        check_and_throw_if_fail(len(hidden_unit_numbers) == len(hidden_unit_activation_functions), "hidden_unit_numbers")
        super(MLPClassifierLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self.output_activation_function = output_activation_function
        self.use_sequence_input = use_sequence_input
        self.norm_inner_output = norm_inner_output
        self.weight_regularizer_batch_norm= weight_regularizer_batch_norm
        self.weight_regularizer_hidden = weight_regularizer_hidden
        self.weight_regularizer_mlp_output = weight_regularizer_mlp_output
        self.input_drop_out_rate = input_drop_out_rate
        self._build_layers()

    def _build_layers(self):
        for hidden_unit_number, hidden_unit_activation_function in zip(self.hidden_unit_numbers, self.hidden_unit_activation_functions):
            drop_out = Dropout(self.input_drop_out_rate)
            self._layers.append(drop_out)
            dense = Dense(hidden_unit_number,
                          activation=hidden_unit_activation_function,
                          W_regularizer =get_copy_of(self.weight_regularizer_hidden),
                          b_regularizer =get_copy_of(self.weight_regularizer_hidden),
                          )
            if self.use_sequence_input:
                dense = TimeDistributed(dense)
            self._layers.append(dense)
            if self.norm_inner_output:
                norm = BatchNormalization(mode=2,
                                          beta_regularizer=get_copy_of(self.weight_regularizer_batch_norm),
                                          gamma_regularizer=get_copy_of(self.weight_regularizer_batch_norm))
                self._layers.append(norm)
        # output layer
        drop_out = Dropout(self.input_drop_out_rate)
        self._layers.append(drop_out)
        dense = Dense(self.output_dim,
                      activation=self.output_activation_function,
                      W_regularizer=get_copy_of(self.weight_regularizer_mlp_output),
                      b_regularizer=get_copy_of(self.weight_regularizer_mlp_output))
        if self.use_sequence_input:
            dense = TimeDistributed(dense)
        self._layers.append(dense)

    def call(self, x, mask=None):
        # apply and then remove mask
        output = remove_mask(x)
        for layer in self._layers:
            output = layer(output)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'use_sequence_input':self.use_sequence_input,
                  'hidden_unit_numbers': self.hidden_unit_numbers,
                  'hidden_unit_activation_functions': self.hidden_unit_activation_functions,
                  'output_activation_function': self.output_activation_function,
                  'norm_inner_output':self.norm_inner_output,
                  'weight_regularizer_hidden': self.weight_regularizer_hidden,
                  'weight_regularizer_mlp_output':self.weight_regularizer_mlp_output,
                  'weight_regularizer_batch_norm':self.weight_regularizer_batch_norm,
                  'input_drop_out_rate': self.input_drop_out_rate,
                  }
        base_config = super(MLPClassifierLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClassifierWithHierarchicalAttention(ComposedLayer):
    def __init__(self, tree_shape, top_level_input_feature_dim, attention_output_dims, attention_weight_vector_dims, embedding_rows,
                 embedding_dim,
                 initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers,
                 hidden_unit_activation_functions, output_activation_function='softmax',
                 use_cnn_as_sequence_to_sequence_encoder=False, input_window_sizes=None, pooling_mode=None,
                 mlp_softmax_classifier_input_drop_out_rate = 0.,
                 hierarchical_layer_dropout_W=0.,
                 hierarchical_layer_dropout_U=0.,
                 attention_input_dropout=0.,
                 attention_dropout=0.,
                 mlp_classifier_norm_inner_output = False, attention_norm_inner_output=False, **kwargs):
        super(ClassifierWithHierarchicalAttention, self).__init__(**kwargs)
        self.tree_shape = tree_shape
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
        self.output_activation_function = output_activation_function
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.pooling_mode = pooling_mode
        self.supports_masking = True
        self.mlp_softmax_classifier_input_drop_out_rate = mlp_softmax_classifier_input_drop_out_rate
        self.hierarchical_layer_dropout_W=hierarchical_layer_dropout_W
        self.hierarchical_layer_dropout_U=hierarchical_layer_dropout_U
        self.attention_input_dropout = attention_input_dropout
        self.attention_dropout = attention_dropout
        self.mlp_classifier_norm_inner_output = mlp_classifier_norm_inner_output
        self.attention_norm_inner_output = attention_norm_inner_output
        self._build_layers()

    def _build_layers(self):
        self.hierarchical_attention = HierarchicalAttention(self.tree_shape,self.top_level_input_feature_dim,
                                                            self.attention_output_dims,
                                                            self.attention_weight_vector_dims,
                                                            self.embedding_rows, self.embedding_dim,
                                                            self.initial_embedding, self.use_sequence_to_vector_encoder,
                                                            self.use_cnn_as_sequence_to_sequence_encoder,
                                                            self.input_window_sizes, self.pooling_mode,
                                                            norm_inner_output=self.attention_norm_inner_output,
                                                            hierarchical_layer_dropout_W=self.hierarchical_layer_dropout_W,
                                                            hierarchical_layer_dropout_U=self.hierarchical_layer_dropout_U,
                                                            attention_input_dropout=self.attention_input_dropout,
                                                            attention_dropout=self.attention_dropout,
                                                          )

        self.mlp_softmax_classifier_input_drop_out = Dropout(self.mlp_softmax_classifier_input_drop_out_rate)

        self.mlp_softmax_classifier = MLPClassifierLayer(self.output_dim, self.hidden_unit_numbers,
                                                         self.hidden_unit_activation_functions,
                                                         self.output_activation_function, norm_inner_output=self.mlp_classifier_norm_inner_output)

        self._layers += [self.hierarchical_attention, self.mlp_softmax_classifier_input_drop_out, self.mlp_softmax_classifier]

    def call(self, inputs, mask=None):
        # shape of output: nb_samples, time_steps, inner_output_dim
        output = self.hierarchical_attention(inputs)
        # drop out
        output = self.mlp_softmax_classifier_input_drop_out(output)
        # shape of output: nb_samples, time_steps, output_dim
        return self.mlp_softmax_classifier(output)

    def compute_mask(self, inputs, mask):
        return self.hierarchical_attention.compute_mask(inputs, mask)

    def get_output_shape_for(self, input_shapes):
        return self.hierarchical_attention.get_output_shape_for(input_shapes)[:-1] + (self.output_dim,)

    def get_config(self):
        config = {'tree_shape': self.tree_shape,
                  'top_level_input_feature_dim': self.top_level_input_feature_dim,
                  'attention_output_dims': self.attention_output_dims,
                  'attention_weight_vector_dims': self.attention_weight_vector_dims,
                  'embedding_rows': self.embedding_rows,
                  'embedding_dim': self.embedding_dim,
                  'initial_embedding': self.initial_embedding,
                  'use_sequence_to_vector_encoder': self.use_sequence_to_vector_encoder,
                  'output_dim': self.output_dim,
                  'hidden_unit_numbers': self.hidden_unit_numbers,
                  'hidden_unit_activation_functions': self.hidden_unit_activation_functions,
                  'output_activation_function': self.output_activation_function,
                  'use_cnn_as_sequence_to_sequence_encoder': self.use_cnn_as_sequence_to_sequence_encoder,
                  'input_window_sizes': self.input_window_sizes,
                  'pooling_mode': self.pooling_mode,
                  'mlp_softmax_classifier_input_drop_out_rate':self.mlp_softmax_classifier_input_drop_out_rate,
                  'mlp_classifier_norm_inner_output':self.mlp_classifier_norm_inner_output,
                  'attention_norm_inner_output': self.attention_norm_inner_output}

        base_config = super(ClassifierWithHierarchicalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras import  constraints

class Embedding(Layer):
    '''Turn positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    This layer can only be used as the first layer in a model.

    # Example

    ```python
      model = Sequential()
      model.add(Embedding(1000, 64, input_length=10))
      # the model will take as input an integer matrix of size (batch, input_length).
      # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
      # now model.output_shape == (None, 10, 64), where None is the batch dimension.

      input_array = np.random.randint(1000, size=(32, 10))

      model.compile('rmsprop', 'mse')
      output_array = model.predict(input_array)
      assert output_array.shape == (32, 10, 64)
    ```

    # Arguments
      input_dim: int > 0. Size of the vocabulary, ie.
          1 + maximum integer index occurring in the input data.
      output_dim: int >= 0. Dimension of the dense embedding.
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      weights: list of Numpy arrays to set as initial weights.
          The list should have 1 element, of shape `(input_dim, output_dim)`.
      W_regularizer: instance of the [regularizers](../regularizers.md) module
        (eg. L1 or L2 regularization), applied to the embedding matrix.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the embedding matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful for [recurrent layers](recurrent.md) which may take
          variable length input. If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
          If mask_zero is set to True, as a consequence, index 0 cannot be
          used in the vocabulary (input_dim should equal |vocabulary| + 2).
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
      dropout: float between 0 and 1. Fraction of the embeddings to drop.

    # Input shape
        2D tensor with shape: `(nb_samples, sequence_length)`.

    # Output shape
        3D tensor with shape: `(nb_samples, sequence_length, output_dim)`.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0.,
                 update_weights=True,
                 learning_rate = None,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout
        self.update_weights = update_weights
        self.W_constraint = constraints.get(W_constraint)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.learning_rate = learning_rate

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((self.input_dim, self.output_dim),
                           name='{}_W'.format(self.name))

        # only when the weights can be updated, allow for constrains and regularization
        if self.update_weights:
            self.trainable_weights = [self.W]
            # attached the customized learning rate
            if self.learning_rate:
                lr = K.variable(self.learning_rate)
                self.W.lr = lr
            self.constraints = {}
            if self.W_constraint:
                self.constraints[self.W] = self.W_constraint

            self.regularizers = []
            if self.W_regularizer:
                self.W_regularizer.set_param(self.W)
                self.regularizers.append(self.W_regularizer)
        else:
            self.non_trainable_weights =[self.W]

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        out = K.gather(W, x)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'dropout': self.dropout,
                  'update_weights': self.update_weights,
                  'learning_rate':self.learning_rate}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
