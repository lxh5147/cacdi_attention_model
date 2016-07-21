'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from keras.layers import Input
from attention_layer import Attention, SequenceToSequenceEncoder, SequenceToVectorEncoder, shape, HierarchicalAttention, MLPClassifierLayer, ClassifierWithHierarchicalAttention
from keras import backend as K
import numpy as np
from attention_exp import  faked_dataset

class AttentionLayerTest(unittest.TestCase):

    def test_attention(self):
        attention = Attention(attention_weight_vector_dim=5)
        input_shape = (3, 5, 10)
        attention.build(input_shape)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        y = attention(x)
        self.assertEqual(shape(y), (3, 10), "y")
        x = Input(shape=(5, 10))
        y = attention(x)
        self.assertEqual(hasattr(y, '_keras_history'), True, "y")

    def test_transform_sequence_to_sequence(self):
        tensor_input = Input(shape=(5, 10))
        output_dim = 20
        sequence_to_sequence_encoder = SequenceToSequenceEncoder(output_dim)
        output_sequence = sequence_to_sequence_encoder(tensor_input)
        self.assertEqual(shape(output_sequence), (None, 5, 40), "output_sequence")
        self.assertEqual(output_sequence._keras_shape, (None, 5, 40), "output_sequence")
        # test with a variable
        input_shape = (3, 5, 10)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_sequence = sequence_to_sequence_encoder(x)
        self.assertEqual(shape(output_sequence), (3, 5, 40), "output_sequence")

    def test_transform_sequence_to_vector_encoder(self):
        output_dim = 20
        sequence_to_vector_encoder = SequenceToVectorEncoder(output_dim)
        tensor_input = Input(shape=(5, 10))
        output_vector = sequence_to_vector_encoder(tensor_input)
        self.assertEqual(shape(output_vector), (None, 20), "output_vector")
        self.assertEqual(output_vector._keras_shape, (None, 20), "output_vector")
        input_shape = (3, 5, 10)
        # test with a variable
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_vector = sequence_to_vector_encoder(x)
        self.assertEqual(shape(output_vector), (3, 20), "output_vector")

    def test_build_hierarchical_attention_layer_inputs(self):
        # time_steps* documents * sections* sentences * words
        input_shape = (7, 8, 5, 6, 9)
        # record, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)

        self.assertEqual(len(inputs) , len(input_feature_dims) + 1, "inputs")
        self.assertEqual(shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs")    # original input
        self.assertEqual(shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs")    # word features
        self.assertEqual(shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs")    # sentence features
        self.assertEqual(shape(inputs[3]), (None, 7, 8, 5, 50), "inputs")    # section features
        self.assertEqual(shape(inputs[4]), (None, 7, 8, 10), "inputs")    # document features
        self.assertEqual(shape(inputs[5]), (None, 7, 20), "inputs")    # snapshot features

    def test_hierarchical_attention_layer_inputs(self):
        # snapshots* documents * sections* sentences * words
        input_shape = (7, 8, 5, 6, 9)
        # snapshot, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        attention_output_dims = (45, 35, 25, 65)
        # document, section, sentence, word
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        embedding_rows = 1024
        embedding_dim = 50
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        hierarchical_attention = HierarchicalAttention(input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder=False)
        output = hierarchical_attention(inputs)
        self.assertEqual(shape(output), (None, 7, 20 + 45 * 2), "output")
        # this is to test the get_output_shape_for method
        self.assertEqual(output._keras_shape, (None, 7, 20 + 45 * 2), "output")
        hierarchical_attention = HierarchicalAttention(input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder=True)
        output = hierarchical_attention(inputs)
        self.assertEqual(shape(output), (None, 7, 390), "output")
        # this is to test the get_output_shape_for method
        self.assertEqual(output._keras_shape, (None, 7, 390), "output")

    def test_mlp_softmax_classifier(self):
        input_shape = (3, 5, 10)
        xval = np.random.random(input_shape) - 0.5
        x = K.variable(xval)
        output_dim = 100
        hidden_unit_numbers = (5, 20)    # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")
        mlp_softmax_classifier = MLPClassifierLayer(output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        y = mlp_softmax_classifier(x)
        self.assertEqual(shape(y), (3, 5, 100), "y")

    def test_hierarchical_attention_model(self):
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
        hidden_unit_activation_functions = ("relu", "relu")
        use_sequence_to_vector_encoder = False

        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))

        classifier = ClassifierWithHierarchicalAttention(input_feature_dims[0], output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        # check inputs of model
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)

        self.assertEqual(len(inputs) , len(input_feature_dims) + 1, "inputs")
        self.assertEqual(shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs")    # original input
        self.assertEqual(shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs")    # word features
        self.assertEqual(shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs")    # sentence features
        self.assertEqual(shape(inputs[3]), (None, 7, 8, 5, 50), "inputs")    # section features
        self.assertEqual(shape(inputs[4]), (None, 7, 8, 10), "inputs")    # document features
        self.assertEqual(shape(inputs[5]), (None, 7, 20), "inputs")    # snapshot features
        # check output
        output = classifier(inputs)
        self.assertEqual(shape(output), (None, 7, 100), "y")

    def test_attention_layer_by_run(self):
        input_shape = (7, 8, 5, 6, 9)
        # record, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        attention_output_dims = (45, 35, 25, 65)
        # document, section, sentence, word
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        embedding_rows = 200
        embedding_dim = 50
        # classifier
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        hierarchical_attention = HierarchicalAttention(input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder=False)
        output = hierarchical_attention(inputs)

        total = 2
        output_dim = 10
        timesteps = input_shape[0]
        x_train, _ = faked_dataset(inputs, total, timesteps, embedding_rows, output_dim)

        # build feed_dic
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[inputs[i]] = x_train[i]
        feed_dict[K.learning_phase()] = 1
        # tf.initialize_all_variables()
        # y_out is fine 2,7, 110
        y_out = K.get_session().run(output, feed_dict=feed_dict)
        self.assertEquals(y_out.shape , (2, 7, 110), "y_out")

    def test_softmax_layer_by_run(self):
        input_sequence = Input(shape=(7, 110))
        output_dim = 5
        hidden_unit_numbers = (5, 20)    # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")
        mlp_softmax_classifier = MLPClassifierLayer(output_dim, hidden_unit_numbers, hidden_unit_activation_functions)

        output = mlp_softmax_classifier(input_sequence)
        # build feed_dic
        total = 4
        feed_dict = {}
        feed_dict[input_sequence] = np.random.random((total,) + shape(input_sequence)[1:])
        feed_dict[K.learning_phase()] = 1
        # tf.initialize_all_variables()
        # y_out is fine 2,7, 110
        y_out = K.get_session().run(output, feed_dict=feed_dict)
        self.assertEqual(y_out.shape , (total, shape(input_sequence)[1], output_dim) , "y_out")

    def test_attention_with_classifier_layer_by_run(self):
        input_shape = (7, 8, 5, 6, 9)
        # record, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        attention_output_dims = (45, 35, 25, 65)
        # document, section, sentence, word
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        embedding_rows = 200
        embedding_dim = 50
        output_dim = 5
        # hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
        # drop_out_rates = ( 0.5, 0.6)
        hidden_unit_numbers = ()    # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ()

        # classifier
        use_sequence_to_vector_encoder = False
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        classifier = ClassifierWithHierarchicalAttention(input_feature_dims[0], attention_output_dims, attention_weight_vector_dims,
                                                         embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder,
                                                         output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        output = classifier(inputs)
        total = 2
        timesteps = input_shape[0]
        x_train, _ = faked_dataset(inputs, total, timesteps, embedding_rows, output_dim)

        # build feed_dic
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[inputs[i]] = x_train[i]
        feed_dict[K.learning_phase()] = 1

        y_out = K.get_session().run(output, feed_dict=feed_dict)
        self.assertEqual(y_out.shape , (total , timesteps, output_dim), "y_out")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
