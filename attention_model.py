'''
Created on Jul 5, 2016

@author: lxh5147
'''


from attention_layer import  HierarchicalAttention, ClassifierWithHierarchicalAttention

from keras.models import Model
import logging

logger = logging.getLogger(__name__)

def build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function,
                                                 use_cnn_as_sequence_to_sequence_encoder = False, input_window_sizes = None, use_pooling_mode = None):
    inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
    classifier = ClassifierWithHierarchicalAttention (input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim,
                 initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function,
                 use_cnn_as_sequence_to_sequence_encoder, input_window_sizes, use_pooling_mode)
    output = classifier(inputs)
    model = Model(input = inputs, output = output)
    return model
