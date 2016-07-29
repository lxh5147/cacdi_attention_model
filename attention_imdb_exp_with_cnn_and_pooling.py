'''
Created on Jul 13, 2016

@author: lxh5147
'''
from attention_model import  build_classifier_with_hierarchical_attention, categorical_crossentropy_ex
from attention_layer import check_and_throw_if_fail
import numpy as np
from keras.callbacks import EarlyStopping
from attention_exp import faked_dataset
from keras.optimizers import SGD

def imdb_exp(max_sentences, max_words, sentence_output_dim, word_output_dim, sentence_attention_weight_vec_dim,
             word_attention_weight_vec_dim, vocabulary_size, word_embedding_dim, initial_embedding, classifier_output_dim, classifier_hidden_unit_numbers, hidden_unit_activation_functions,
             use_cnn_as_sequence_to_sequence_encoder = False, input_window_sizes = None, use_max_pooling_as_attention = False):

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

    model = build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding,
                                                         use_sequence_to_vector_encoder, classifier_output_dim, classifier_hidden_unit_numbers, hidden_unit_activation_functions,
                                                         use_cnn_as_sequence_to_sequence_encoder, input_window_sizes , use_max_pooling_as_attention)
    # compile the model
    model.compile(optimizer = SGD(momentum = 0.9)   , loss = categorical_crossentropy_ex, metrics = ['accuracy'])
    # train
    total = 10
    batch_size = 2
    nb_epoch = 5
    x_train, y_train = faked_dataset(model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)

    model.fit(x_train, y_train, batch_size, nb_epoch, verbose = 1, callbacks = [EarlyStopping(patience = 5)],
            validation_split = 0.1, validation_data = None, shuffle = True,
            class_weight = None, sample_weight = None)
    # evaluate
    total = 4
    x_test, y_test = faked_dataset(model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)
    model.evaluate(x_test, y_test, batch_size = 2, verbose = 1, sample_weight = None)
    # predict
    total = 2
    x_pred, _ = faked_dataset(model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)
    y_pred = model.predict(x_pred, batch_size = 1, verbose = 1)
    check_and_throw_if_fail(y_pred.shape == (total, timesteps, classifier_output_dim), "y_pred")



if __name__ == '__main__':
    # http://aclweb.org/anthology/N16-1174
    # max_sentences = 10
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
    use_cnn_as_sequence_to_sequence_encoder = True
    # sentence, word
    input_window_sizes = [3, 2],
    use_max_pooling_as_attention = True
    # batch size = 64
    imdb_exp(max_sentences, max_words, sentence_output_dim, word_output_dim, sentence_attention_weight_vec_dim, word_attention_weight_vec_dim, vocabulary_size, word_embedding_dim, initial_embedding, classifier_output_dim, classifier_hidden_unit_numbers, hidden_unit_activation_functions,
             use_cnn_as_sequence_to_sequence_encoder, input_window_sizes, use_max_pooling_as_attention)

