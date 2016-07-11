'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from attention_model import build_hierarchical_attention_layers
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
    

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()