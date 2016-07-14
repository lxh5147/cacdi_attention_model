'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from keras.layers import Input
from attention_layer import Attention, SequenceToSequenceEncoder,SequenceToVectorEncoder,HierarchicalAttention
from keras import backend as K
import numpy as np

class AttentionLayerTest(unittest.TestCase):

    def test_attention(self):
        attention = Attention(attention_weight_vector_dim=5)
        input_shape=(3,5,10)
        attention.build(input_shape)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)        
        y = attention(x)
        self.assertEqual(K.int_shape(y), (3,10), "y")
        x = Input(shape=(5,10))
        y = attention(x)
        self.assertEqual( hasattr(y, '_keras_history'),True, "y")

    def test_transform_sequence_to_sequence(self):
        tensor_input = Input(shape=(5,10))
        output_dim = 20
        sequence_to_sequence_encoder = SequenceToSequenceEncoder(output_dim)
        output_sequence= sequence_to_sequence_encoder(tensor_input)
        self.assertEqual(K.int_shape(output_sequence), (None,5,40), "output_sequence")
        self.assertEqual(output_sequence._keras_shape, (None,5,40), "output_sequence")
        #test with a variable
        input_shape=(3,5,10)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_sequence= sequence_to_sequence_encoder(x)
        self.assertEqual(K.int_shape(output_sequence), (3,5,40), "output_sequence")
        
    def test_transform_sequence_to_vector_encoder(self):
        output_dim = 20
        sequence_to_vector_encoder = SequenceToVectorEncoder(output_dim)
        tensor_input = Input(shape=(5,10))
        output_vector= sequence_to_vector_encoder(tensor_input)
        self.assertEqual(K.int_shape(output_vector), (None,20), "output_vector")
        self.assertEqual(output_vector._keras_shape, (None,20), "output_vector")
        input_shape=(3,5,10)
        #test with a variable
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_vector= sequence_to_vector_encoder(x)
        self.assertEqual(K.int_shape(output_vector), (3,20), "output_vector")
        
    def test_build_hierarchical_attention_layer_inputs(self):
        # time_steps* documents * sections* sentences * words
        input_shape=(7,8,5,6,9)
        #record, document,section,sentence,word
        input_feature_dims=(20,10,50,60,30)
        #document, section, sentence, word
        inputs= HierarchicalAttention.build_inputs(input_shape, input_feature_dims)

        self.assertEqual(len(inputs) , len(input_feature_dims)+1, "inputs")
        self.assertEqual( K.int_shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs") #original input
        self.assertEqual( K.int_shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs") # word features
        self.assertEqual( K.int_shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs") # sentence features
        self.assertEqual( K.int_shape(inputs[3]), (None, 7, 8, 5, 50), "inputs") #section features
        self.assertEqual( K.int_shape(inputs[4]), (None, 7, 8, 10), "inputs") #document features
        self.assertEqual( K.int_shape(inputs[5]), (None, 7, 20), "inputs") #snapshot features
        
    def test_hierarchical_attention_layer_inputs(self):
        # snapshots* documents * sections* sentences * words
        input_shape=(7,8,5,6,9)
        #snapshot, document,section,sentence,word
        input_feature_dims=(20,10,50,60,30)
        #document, section, sentence, word
        attention_output_dims=(45,35,25,65)
        #document, section, sentence, word
        attention_weight_vector_dims = (82,72,62,52)
        #embedding
        embedding_rows = 1024
        embedding_dim = 50
        initial_embedding = np.random.random((embedding_rows,embedding_dim)) 
        inputs= HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        hierarchical_attention = HierarchicalAttention(attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder = False)
        output = hierarchical_attention(inputs)
        self.assertEqual(K.int_shape(output), (None,7,20+45*2), "output")
        #this is to test the get_output_shape_for method
        self.assertEqual(output._keras_shape, (None,7,20+45*2), "output")        
        hierarchical_attention = HierarchicalAttention(attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder = True)
        output = hierarchical_attention(inputs)
        self.assertEqual(K.int_shape(output), (None,7,390), "output")
        #this is to test the get_output_shape_for method      
        self.assertEqual(output._keras_shape, (None,7,390), "output")        
         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()