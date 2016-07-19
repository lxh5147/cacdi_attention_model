'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from attention_model import  apply_mlp_softmax_classifier, build_classifier_with_hierarchical_attention
from attention_layer import shape
from keras import backend as K
import numpy as np

class AttentionModelTest(unittest.TestCase):

    def test_apply_mlp_softmax_classifier(self):
        input_shape = (3, 5, 10)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_dim = 100
        hidden_unit_numbers = (5, 20)    # 5--> first hidden layer, 20 --> second hidden layer
        drop_out_rates = (0.5, 0.6)
        y = apply_mlp_softmax_classifier(x, output_dim, hidden_unit_numbers, drop_out_rates)
        self.assertEqual(shape(y), (3, 5, 100), "y")

    def test_build_hierarchical_attention_model(self):
        # time_steps* documents * sections* sentences * words
        input_shape = (7, 8, 5, 6, 9)
        # record, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        output_dims = (45, 35, 25, 65)
        # document, section, sentence, word
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        vocabulary_size = 1024
        word_embedding_dim = 50
        # classifier
        output_dim = 100
        hidden_unit_numbers = (5, 20)    # 5--> first hidden layer, 20 --> second hidden layer
        drop_out_rates = (0.5, 0.6)
        use_sequence_to_vector_encoder = False

        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
        model = build_classifier_with_hierarchical_attention(input_feature_dims[0], input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, drop_out_rates)
        # check inputs of model
        inputs = model.inputs
        self.assertEqual(len(inputs) , len(input_feature_dims) + 1, "inputs")
        self.assertEqual(shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs")    # original input
        self.assertEqual(shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs")    # word features
        self.assertEqual(shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs")    # sentence features
        self.assertEqual(shape(inputs[3]), (None, 7, 8, 5, 50), "inputs")    # section features
        self.assertEqual(shape(inputs[4]), (None, 7, 8, 10), "inputs")    # document features
        self.assertEqual(shape(inputs[5]), (None, 7, 20), "inputs")    # snapshot features
        # check output of model
        self.assertEqual(len(model.outputs), 1, "model")
        self.assertEqual(shape(model.outputs[0]), (None, 7, 100), "y")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
