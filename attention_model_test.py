'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from attention_model import build_hierarchical_attention_layers, apply_mlp_softmax_classifier, build_hierarchical_attention_model
from keras import backend as K
import numpy as np

class AttentionModelTest(unittest.TestCase):

    def test_build_hierarchical_attention_layers(self):
        # time_steps* documents * sections* sentences * words
        input_shape=(7,8,5,6,9)
        #record, document,section,sentence,word
        input_feature_dims=(20,10,50,60,30)
        #document, section, sentence, word
        output_dims=(45,35,25,65)
        #document, section, sentence, word
        attention_weight_vector_dims = (82,72,62,52)
        #embedding
        vocabulary_size = 1024
        word_embedding_dim = 50 
        initial_embedding = np.random.random((vocabulary_size,word_embedding_dim)) 
        
        inputs, output_attention= build_hierarchical_attention_layers(input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding)

        self.assertEqual(len(inputs) , len(input_feature_dims)+1, "inputs")
        self.assertEqual( K.int_shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs") #original input
        self.assertEqual( K.int_shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs") # word features
        self.assertEqual( K.int_shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs") # sentence features
        self.assertEqual( K.int_shape(inputs[3]), (None, 7, 8, 5, 50), "inputs") #section features
        self.assertEqual( K.int_shape(inputs[4]), (None, 7, 8, 10), "inputs") #document features
        self.assertEqual( K.int_shape(inputs[5]), (None, 7, 20), "inputs") #snapshot features
        self.assertEqual(K.int_shape(output_attention), (None,7,110), "output_attention")
    
    def test_apply_mlp_softmax_classifier(self):
        input_shape=(3,5,10)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_dim = 100
        hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
        drop_out_rates = ( 0.5, 0.6)    
        y = apply_mlp_softmax_classifier(x, output_dim, hidden_unit_numbers, drop_out_rates)
        self.assertEqual(K.int_shape(y), (3*5,100), "y")
        
    def test_build_hierarchical_attention_model(self):
        # time_steps* documents * sections* sentences * words
        input_shape=(7,8,5,6,9)
        #record, document,section,sentence,word
        input_feature_dims=(20,10,50,60,30)
        #document, section, sentence, word
        output_dims=(45,35,25,65)
        #document, section, sentence, word
        attention_weight_vector_dims = (82,72,62,52)
        #embedding
        vocabulary_size = 1024
        word_embedding_dim = 50
        #classifier
        output_dim = 100
        hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
        drop_out_rates = ( 0.5, 0.6) 
        initial_embedding = np.random.random((vocabulary_size,word_embedding_dim))
        model = build_hierarchical_attention_model(input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding, output_dim, hidden_unit_numbers, drop_out_rates)
        #check inputs of model
        inputs=model.inputs
        self.assertEqual(len(inputs) , len(input_feature_dims)+1, "inputs")
        self.assertEqual( K.int_shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs") #original input
        self.assertEqual( K.int_shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs") # word features
        self.assertEqual( K.int_shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs") # sentence features
        self.assertEqual( K.int_shape(inputs[3]), (None, 7, 8, 5, 50), "inputs") #section features
        self.assertEqual( K.int_shape(inputs[4]), (None, 7, 8, 10), "inputs") #document features
        self.assertEqual( K.int_shape(inputs[5]), (None, 7, 20), "inputs") #snapshot features
        #check outoput of model
        self.assertEqual(len(model.outputs), 1, "model")
        self.assertEqual(K.int_shape(model.outputs[0]), (3*5,100), "y")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()