'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from attention_model import   build_classifier_with_hierarchical_attention
from attention_layer import shape
import numpy as np
from keras.optimizers import SGD


import os
from keras.models import model_from_yaml
from attention_layer import ClassifierWithHierarchicalAttention, MaskSequence
from attention_exp import  faked_dataset
from keras.callbacks import EarlyStopping

class AttentionModelTest(unittest.TestCase):

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
        hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")
        use_sequence_to_vector_encoder = False

        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
        model = build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        # check inputs of model
        inputs = model.inputs
        self.assertEqual(len(inputs) , len(input_feature_dims) + 1, "inputs")
        self.assertEqual(shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs")  # original input
        self.assertEqual(shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs")  # word features
        self.assertEqual(shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs")  # sentence features
        self.assertEqual(shape(inputs[3]), (None, 7, 8, 5, 50), "inputs")  # section features
        self.assertEqual(shape(inputs[4]), (None, 7, 8, 10), "inputs")  # document features
        self.assertEqual(shape(inputs[5]), (None, 7, 20), "inputs")  # snapshot features
        # check output of model
        self.assertEqual(len(model.outputs), 1, "model")
        self.assertEqual(shape(model.outputs[0]), (None, 7, 100), "y")

    def test_model_save_load(self):
        max_sentences = None
        # max_words = 15
        max_words = None

        sentence_output_dim = 50
        word_output_dim = 50

        sentence_attention_weight_vec_dim = 50
        word_attention_weight_vec_dim = 50

        vocabulary_size = 100
        word_embedding_dim = 200
        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
        classifier_output_dim = 20
        classifier_hidden_unit_numbers = []
        hidden_unit_activation_functions = []
        use_cnn_as_sequence_to_sequence_encoder = False
        # sentence, word
        input_window_sizes = [3, 2]
        pooling_mode = "max"
        timesteps = 1
        # time_steps*  sentences * words
        input_shape = (timesteps, max_sentences, max_words)
        # comment,sentence,word
        input_feature_dims = (0, 0, 0)
        # sentence, word
        output_dims = (sentence_output_dim, word_output_dim)
        # sentence, word
        attention_weight_vector_dims = (sentence_attention_weight_vec_dim, word_attention_weight_vec_dim)
        # embedding
        # classifier
        use_sequence_to_vector_encoder = False

        output_activation_function = 'softmax'

        model = build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, output_dims,
                                                             attention_weight_vector_dims, vocabulary_size,
                                                             word_embedding_dim, initial_embedding,
                                                             use_sequence_to_vector_encoder, classifier_output_dim,
                                                             classifier_hidden_unit_numbers,
                                                             hidden_unit_activation_functions,
                                                             output_activation_function,
                                                             use_cnn_as_sequence_to_sequence_encoder,
                                                             input_window_sizes, pooling_mode)
        # compile the model
        model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        total = 4
        batch_size = 2
        nb_epoch = 2
        x_train, y_train = faked_dataset(
            model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)

        model.fit(
            x_train, y_train, batch_size, nb_epoch, verbose=1,
            callbacks=[EarlyStopping(patience=5)], validation_split=0.1,
            validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None)

        # predict
        total = 2
        x_pred, _ = faked_dataset(
            model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)

        y_pred = model.predict(x_pred, batch_size=1, verbose=1)

        # save and load model weights
        model_weights_file = "my_model_weights.h5"
        model.save_weights(model_weights_file,overwrite=True)
        model_config_file="my_model.json"
        yaml = model.to_yaml()
        # save the yaml
        yaml_file = "my_model.yaml"
        with open(yaml_file, 'w') as file_:
            file_.write(yaml)
        custom_objects={'ClassifierWithHierarchicalAttention': ClassifierWithHierarchicalAttention,
                        'MaskSequence': MaskSequence}
        with open(yaml_file, 'r') as file_:
            restored_yaml = file_.read()
        restored_model = model_from_yaml(restored_yaml,custom_objects) #just created layers, but not restored connections between layers
        restored_model.load_weights(model_weights_file)
        # restored_model is not compiled, comple the model to completely restore the model
        restored_model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        # run prediction with the restored model
        y_pred_restored =  restored_model.predict(x_pred, batch_size=1, verbose=1)
        self.assertTrue(np.array_equal(y_pred,y_pred_restored),"model_save_restore")
        os.remove(model_weights_file)
        os.remove(yaml_file)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
