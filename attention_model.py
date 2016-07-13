'''
Created on Jul 5, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.layers import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from attention_layer import check_and_throw_if_fail, HierarchicalAttention 
from keras.models import Model
import logging

logger = logging.getLogger(__name__)


def apply_mlp_softmax_classifier(input_sequence, output_dim, hidden_unit_numbers, drop_out_rates):
    '''
    input_sequence: input sequence, batch_size * time_steps * input_dim
    hidden_unit_numbers: number of hidden units of each hidden layer
    drop_out_rates: drop out rates of each hidden layer
    output_dim: output dimension
    returns a tensor of shape: batch_size*time_steps*output_dim
    '''
    check_and_throw_if_fail(K.ndim(input_sequence) == 3 , "input_sequence")
    output = input_sequence
    for hidden_unit_number, drop_out_rate in zip(hidden_unit_numbers, drop_out_rates):
        output = TimeDistributed(Dense(hidden_unit_number, init = 'uniform'))(output)
        output = TimeDistributed(Activation('tanh'))(output)
        output = TimeDistributed(Dropout(drop_out_rate))(output)
    output = TimeDistributed(Dense(output_dim, init = 'uniform'))(output)
    output = TimeDistributed(Activation('softmax'))(output)
    return output

# 
def build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, drop_out_rates):
    inputs= HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
    hierarchical_attention = HierarchicalAttention(attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder)
    output = hierarchical_attention(inputs)
    output = apply_mlp_softmax_classifier(output, output_dim, hidden_unit_numbers, drop_out_rates)
    model = Model(input = inputs, output = output)
    return model
