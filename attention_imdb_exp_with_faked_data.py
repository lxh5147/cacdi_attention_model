'''
Created on Jul 13, 2016

@author: lxh5147
'''
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

from attention_exp import faked_dataset
from attention_layer import check_and_throw_if_fail
from attention_model import build_classifier_with_hierarchical_attention


np.random.seed(1321)

# x_train, y_train, x_test, y_test, x_pred, batch_size, nb_epoch,
def imdb_exp(max_sentences,
             max_words,
             sentence_output_dim,
             word_output_dim,
             sentence_attention_weight_vec_dim,
             word_attention_weight_vec_dim,
             vocabulary_size,
             word_embedding_dim,
             initial_embedding,
             classifier_output_dim,
             classifier_hidden_unit_numbers,
             hidden_unit_activation_functions,
             output_activation_function):

    timesteps = 1
    # time_steps*  sentences * words
    input_shape = (timesteps, max_sentences, max_words)
    # comment,sentence,word
    input_feature_dims = (0, 0, 0)
    # sentence, word
    output_dims = (sentence_output_dim, word_output_dim)
    # sentence, word
    attention_weight_vector_dims = (sentence_attention_weight_vec_dim,
                                    word_attention_weight_vec_dim)
    # embedding
    # classifier
    use_sequence_to_vector_encoder = False

    model = build_classifier_with_hierarchical_attention(
        input_shape,
        input_feature_dims,
        output_dims,
        attention_weight_vector_dims,
        vocabulary_size,
        word_embedding_dim,
        initial_embedding,
        use_sequence_to_vector_encoder,
        classifier_output_dim,
        classifier_hidden_unit_numbers,
        hidden_unit_activation_functions,
        output_activation_function,
        attention_norm_inner_output=True)

    # compile the model
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9, decay=0.01),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

    # train
    total = 10
    batch_size = 2
    nb_epoch = 5
    x_train, y_train = faked_dataset(
        model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)

    filepath = "imdb_faked_model.weights.{epoch:03d}.h5"
    train_history = model.fit(
        x_train, y_train, batch_size, nb_epoch, verbose = 1,
        callbacks = [EarlyStopping(patience = 5), ModelCheckpoint(filepath=filepath, save_weights_only=True)], validation_split = 0.1,
        validation_data = None, shuffle = False, class_weight = None,
        sample_weight = None)

    # restore weights corresponding to the lowest val_loss
    val_loss = train_history.history['val_loss']
    best_epoch = np.argmin( val_loss )
    model.load_weights(filepath.format(epoch=best_epoch))

    # evaluate
    total = 4
    x_test, y_test = faked_dataset(
        model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)
    evaluation_results = model.evaluate(x_test, y_test, batch_size = 2, verbose = 1, sample_weight = None)
    print("evaluation results:")
    print(evaluation_results)
    # predict
    total = 2
    x_pred, _ = faked_dataset(
        model.inputs, total, timesteps, vocabulary_size, classifier_output_dim)
    y_pred = model.predict(x_pred, batch_size = 1, verbose = 1)
    check_and_throw_if_fail(
        y_pred.shape == (total, timesteps, classifier_output_dim), "y_pred")

if __name__ == '__main__':
    # http://aclweb.org/anthology/N16-1174
    # max_sentences = 10
    max_sentences = None
    # max_words = 15
    max_words = None

    sentence_output_dim = 50
    word_output_dim = 50

    sentence_attention_weight_vec_dim = 5
    word_attention_weight_vec_dim = 8

    vocabulary_size = 100
    word_embedding_dim = 200
    initial_embedding = np.random.random((vocabulary_size, word_embedding_dim))
    classifier_output_dim = 20
    classifier_hidden_unit_numbers = []
    hidden_unit_activation_functions = []
    output_activation_function="sigmoid"
    # batch size = 64
    imdb_exp(
        max_sentences,
        max_words,
        sentence_output_dim,
        word_output_dim,
        sentence_attention_weight_vec_dim,
        word_attention_weight_vec_dim,
        vocabulary_size,
        word_embedding_dim,
        initial_embedding,
        classifier_output_dim,
        classifier_hidden_unit_numbers,
        hidden_unit_activation_functions,
        output_activation_function)