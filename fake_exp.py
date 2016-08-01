'''
Created on Jul 13, 2016

@author: lxh5147
'''
from attention_model import  build_classifier_with_hierarchical_attention, binary_crossentropy_ex
from attention_layer import check_and_throw_if_fail
from attention_exp import  faked_dataset
import numpy as np
from keras.callbacks import EarlyStopping

def faked_exp():
    # time_steps* documents * sections* sentences * words
    input_shape = (7, 8, 5, 6, 9)
    # record, document,section,sentence,word
    input_feature_dims = (20, 10, 50, 60, 30)
    # document, section, sentence, word
    output_dims = (45, 35, 25, 65)
    # document, section, sentence, word
    attention_weight_vector_dims = (82, 72, 62, 52)
    # embedding
    vocabulary_size = 200
    word_embedding_dim = 50
    # classifier
    output_dim = 5
    hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
    hidden_unit_activation_functions = ("relu", "relu")
    use_sequence_to_vector_encoder = False

    initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
    model = build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, output_dims,
                                                          attention_weight_vector_dims, vocabulary_size, word_embedding_dim,
                                                          initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function = 'sigmoid')
    # compile the model
    model.compile(optimizer = 'rmsprop', loss = binary_crossentropy_ex, metrics = ['accuracy'])
    # train
    total = 10
    batch_size = 2
    nb_epoch = 5
    timesteps = input_shape[0]

    x_train, y_train = faked_dataset(model.inputs, total, timesteps, vocabulary_size, output_dim)
    model.fit(x_train, y_train, batch_size, nb_epoch, verbose = 1,
              callbacks = [EarlyStopping(patience = 5)], validation_split = 0.1, validation_data = None, shuffle = True,
              class_weight = None, sample_weight = None)
    # evaluate
    total = 4
    x_test, y_test = faked_dataset(model.inputs, total, timesteps, vocabulary_size, output_dim)
    model.evaluate(x_test, y_test, batch_size = 2, verbose = 1, sample_weight = None)
    # predict
    total = 2
    x_pred, _ = faked_dataset(model.inputs, total, timesteps, vocabulary_size, output_dim)
    y_pred = model.predict(x_pred, batch_size = 1, verbose = 1)
    check_and_throw_if_fail(y_pred.shape == (total, timesteps, output_dim), "y_pred")

if __name__ == '__main__':
    faked_exp()
