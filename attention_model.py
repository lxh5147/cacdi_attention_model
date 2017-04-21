'''
Created on Jul 5, 2016

@author: lxh5147
'''


from attention_layer import  HierarchicalAttention, mask_sequence,ClassifierWithHierarchicalAttention, MaskSequence
from keras.models import Model


def build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function="softmax",
                                                 use_cnn_as_sequence_to_sequence_encoder = False, input_window_sizes = None, pooling_mode = None, mlp_softmax_classifier_input_drop_out=0.,
                                                 hierarchical_layer_dropout_W=0.,
                                                 hierarchical_layer_dropout_U=0.,
                                                 attention_context_dropout=0.,
                                                 attention_dropout=0.,
                                                 attention_input_dropout=0.,
                                                 mlp_classifier_norm_inner_output=False,
                                                 attention_norm_inner_output = False,
                                                 input_mask_value = 1,
                                                 apply_input_mask = False):
    # prepare inputs with mask if required
    inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
    if apply_input_mask:
        # attach masks for inputs, calculated using mask_value
        inputs_with_masks = [mask_sequence(inputs[0], mask_value=input_mask_value)] + \
                 [mask_sequence(input, mask_value=0, is_vector_sequence= True)  for input in inputs[1:]]
    else:
        inputs_with_masks = inputs

    classifier = ClassifierWithHierarchicalAttention (input_shape,input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim,
                                                      initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function,
                                                      use_cnn_as_sequence_to_sequence_encoder, input_window_sizes, pooling_mode,mlp_softmax_classifier_input_drop_out,
                                                      hierarchical_layer_dropout_W=hierarchical_layer_dropout_W,
                                                      hierarchical_layer_dropout_U=hierarchical_layer_dropout_U,
                                                      attention_context_dropout=attention_context_dropout,
                                                      attention_dropout=attention_dropout,
                                                      attention_input_dropout=attention_input_dropout,
                                                      mlp_classifier_norm_inner_output=mlp_classifier_norm_inner_output,
                                                      attention_norm_inner_output = attention_norm_inner_output)
    # this will apply the masks associated with the inputs automatically
    output = classifier(inputs_with_masks)
    # will automatically apply output mask to the output
    model = Model(input = inputs, output = output)
    return model
