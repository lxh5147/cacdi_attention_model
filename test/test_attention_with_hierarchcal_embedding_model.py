'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from HierarchicalEmbeddingWithAttentionLayer import (
    build_classifier_with_hierarchical_embedding_attention,
    ClassifierWithHierarchicalEmbeddingAttention)

from engine import ModelEx as Model

from attention_layer import shape
import numpy as np
from keras.optimizers import SGD


import os
from keras.models import model_from_yaml
from attention_layer import MaskSequence

from attention_exp import (fake_data,to_binary_matrix)

def fake_dataset(total,input_feature_dims,  output_dim):
    sequence_inputs = []
    sparse_feature_inputs =[]

    total_sequence_input = len(input_feature_dims)-1
    # nb
    for i,input_feature_dim in enumerate(input_feature_dims[:-1]):
        # for the last sequence input, its nb_samples = total
        if i == total_sequence_input -1:
            nb_samples = total
        else:
            nb_samples = 20
        sequence_input = fake_data([nb_samples,5],
                                    'int32',
                                    10) #small enough vocabulary
        sequence_inputs.append(sequence_input)
        sparse_feature_input = fake_data(
                                    [nb_samples,
                                     5,
                                     input_feature_dim],
                                    )
        sparse_feature_inputs.append(sparse_feature_input)

    sparse_feature_input = fake_data(
                (total,
                input_feature_dims[-1]))

    sparse_feature_inputs.append(sparse_feature_input)

    #output
    output_val = fake_data((total,),
                           'int32',
                           max_int=output_dim)
    output_val = to_binary_matrix(output_val,
                                  max_int=output_dim)

    # reverse the order:
    sequence_inputs.reverse()
    sparse_feature_inputs.reverse()

    return sequence_inputs + sparse_feature_inputs,\
           output_val

class HierarchicalEmbeddingWithAttentionTest(unittest.TestCase):

    def test_build_model(self):
        # word, ..., snapshot
        input_feature_dims = (20, 10, 50, 60, 30)
        # words, ..., documents
        output_dims = (45, 35, 25, 65)
        # words, ..., documents
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        vocabulary_size = 1024
        word_embedding_dim = 50
        # classifier
        output_dim = 100
        hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")

        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
        model = build_classifier_with_hierarchical_embedding_attention(
            input_feature_dims,
            output_dims,
            attention_weight_vector_dims,
            vocabulary_size,
            word_embedding_dim,
            initial_embedding,
            output_dim,
            hidden_unit_numbers,
            hidden_unit_activation_functions)
        # check inputs of model
        inputs = model.inputs
        self.assertEqual(len(inputs),9,'inputs')

        sequence_inputs = inputs[:4]
        sequence_inputs.reverse()
        sparse_inputs = inputs[4:]
        sparse_inputs.reverse()

        for i in xrange(4):
            self.assertEqual(shape(sequence_inputs[i]),(None,None))
        for i in xrange(4):
            self.assertEqual(shape(sparse_inputs[i]),
                             (None,
                              None,
                              input_feature_dims[i]))
        # last sparse feature input tensor
        self.assertEqual(shape(sparse_inputs[4]),
                         (None, input_feature_dims[4]))

        self.assertEqual(shape(model.outputs[0]), (None,  100))

    def test_model_train(self):
        # word, ..., snapshot
        input_feature_dims = (20, 10, 50, 60, 30)
        # words, ..., documents
        output_dims = (45, 35, 25, 65)
        # words, ..., documents
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        vocabulary_size = 1024
        word_embedding_dim = 50
        # classifier
        output_dim = 100
        hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")
        from keras.regularizers import l2,WeightRegularizer
        weight_reg = l2().get_config()

        initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
        model = build_classifier_with_hierarchical_embedding_attention(
            input_feature_dims,
            output_dims,
            attention_weight_vector_dims,
            vocabulary_size,
            word_embedding_dim,
            initial_embedding,
            output_dim,
            hidden_unit_numbers,
            hidden_unit_activation_functions,
            weight_regularizer=weight_reg)

        # compile the model
        model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        regs = model.regularizers
        weights = model.trainable_weights
        reg_parameters=[]
        for reg in regs:
            self.assertTrue(isinstance(reg,WeightRegularizer),'reg')
            reg_parameters.append(reg.p)
        # assert two sets equals
        self.assertEqual(len(reg_parameters),len(weights))
        for p in reg_parameters:
            self.assertTrue(p in weights)

        total = 4
        x_train, y_train = fake_dataset(
            total,
            input_feature_dims,
            output_dim)

        model.train_on_batch(
            x_train,
            y_train)

        # predict
        total = 2
        x_pred, _ = fake_dataset(
            total,
            input_feature_dims,
            output_dim)

        y_pred = model.predict_on_batch(x_pred)

        # save and load model weights
        model_weights_file = "my_model_embedding_weights.h5"
        model.save_weights(model_weights_file,overwrite=True)
        yaml = model.to_yaml()
        # save the yaml
        yaml_file = "my_model.yaml"
        with open(yaml_file, 'w') as file_:
            file_.write(yaml)
        custom_objects={'ClassifierWithHierarchicalEmbeddingAttention': ClassifierWithHierarchicalEmbeddingAttention,
                        'MaskSequence': MaskSequence,
                        'ModelEx':Model}
        with open(yaml_file, 'r') as file_:
            restored_yaml = file_.read()
        restored_model = model_from_yaml(restored_yaml,custom_objects) #just created layers, but not restored connections between layers
        restored_model.load_weights(model_weights_file)
        # restored_model is not compiled, comple the model to completely restore the model
        restored_model.compile(optimizer=SGD(momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        # run prediction with the restored model
        y_pred_restored =  restored_model.predict_on_batch(x_pred)
        self.assertTrue(np.array_equal(y_pred,y_pred_restored),"model_save_restore")

        os.remove(model_weights_file)
        os.remove(yaml_file)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
