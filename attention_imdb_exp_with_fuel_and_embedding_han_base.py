'''
Created on Jul 13, 2016

@author: lxh5147
'''
import functools
import logging
import os.path
import numpy as np

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from fuel.transformers import  FilterSources

from fuel_cacdi.datasets.imdb import IMDBDatasetWrapper

from HierarchicalEmbeddingWithAttentionLayer import  build_classifier_with_hierarchical_embedding_attention
from tree_tensor_converter import get_sequence_inputs

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # INFO)

source_dir = os.getenv('IMDB_SOURCE_DIR', '/hcnlp/users/xiaohua_liu/data/imdb/hierarchical_core_nlp_fresh')

train_source = os.path.join(source_dir, "train")
valid_source = os.path.join(source_dir, "valid")
test_source = os.path.join(source_dir, "test")

partition_names = ["train", "valid", "test"]
partition_sources = [train_source, valid_source, test_source]

pretrained_word_embeddings_path = os.path.join(
    source_dir, "train_valid_set_w2v.npy")

current_row = None

# x_train, y_train, x_test, y_test, x_pred, batch_size, nb_epoch,
def imdb_exp(datasets, 
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
             optimizer, 
             use_cnn_as_sequence_to_sequence_encoder=False,
             input_window_sizes=None,
             pooling_mode = None,
             save_weights_for_each_epoch=True,
             mlp_softmax_classifier_input_drop_out = 0.,
             hierarchical_layer_dropout_W=0.,
             hierarchical_layer_dropout_U=0.,
             attention_dropout=0.,
             attention_input_dropout=0.,
             mlp_classifier_norm_inner_output = False,
             attention_norm_inner_output = False,
             input_mask_value = -1,
             weights_file_path_prefix= '',
             unpacked = False,
             initial_model_weights_file_path = None,
             ):

    if initial_model_weights_file_path:
        #check if model weights file exist
        if not os.path.exists(initial_model_weights_file_path):
            print("Initial model weights file does not exist: %s" % initial_model_weights_file_path)
            return
        if not os.path.isfile(initial_model_weights_file_path):
            print("Initial model weights file is a directory, should be a file: %s" % initial_model_weights_file_path)
            return

    if unpacked:
        # comment,sentence,word
        input_feature_dims = (0,0)
        # sentence, word
        output_dims = ( word_output_dim,)
        # sentence, word
        attention_weight_vector_dims = (word_attention_weight_vec_dim,)
    else:
        # comment,sentence,word
        input_feature_dims = (0, 0, 0)
        # sentence, word
        output_dims = (sentence_output_dim, word_output_dim)
        # sentence, word
        attention_weight_vector_dims = (sentence_attention_weight_vec_dim,
                                        word_attention_weight_vec_dim)

    model = build_classifier_with_hierarchical_embedding_attention(
        input_feature_dims,
        output_dims, 
        attention_weight_vector_dims, 
        vocabulary_size, 
        word_embedding_dim, 
        initial_embedding,
        classifier_output_dim, 
        classifier_hidden_unit_numbers, 
        hidden_unit_activation_functions, 
        output_activation_function="softmax",
        use_cnn_as_sequence_to_sequence_encoder= use_cnn_as_sequence_to_sequence_encoder,
        input_window_sizes=input_window_sizes,
        pooling_mode=pooling_mode,
        mlp_softmax_classifier_input_drop_out=mlp_softmax_classifier_input_drop_out,
        hierarchical_layer_dropout_W=hierarchical_layer_dropout_W,
        hierarchical_layer_dropout_U=hierarchical_layer_dropout_U,
        attention_dropout=attention_dropout,
        attention_input_dropout=attention_input_dropout,
        mlp_classifier_norm_inner_output = mlp_classifier_norm_inner_output,
        attention_norm_inner_output = attention_norm_inner_output,
        input_mask_value = input_mask_value)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # pre-load weights, the caller should ensure that the weights file is valid for this model
    if initial_model_weights_file_path:
        logger.info("Initialize model weights by loading weights from %s" % initial_model_weights_file_path)
        model.load_weights(initial_model_weights_file_path)

    # train
    nb_epoch = 100
    weights_file_path = weights_file_path_prefix + "imdb_exp"
    yaml_file = weights_file_path_prefix + "imdb_exp"
    if use_cnn_as_sequence_to_sequence_encoder:
        weights_file_path += "_cnn"
        yaml_file += "_cnn"
    if pooling_mode:
        weights_file_path += "_" + pooling_mode
        yaml_file += "_" + pooling_mode
    weights_file_path += "_" + optimizer.__class__.__name__

    # consider learning rate,decay and momentum as part of the weight file
    if hasattr(optimizer,'lr'):
        weights_file_path += "_lr_" + str(round(K.get_value(getattr(optimizer, 'lr')), 6))
    if hasattr(optimizer, 'decay'):
        weights_file_path += "_decay_" + str(round(K.get_value(getattr(optimizer, 'decay')), 6))
    if hasattr(optimizer, 'momentum'):
        weights_file_path += "_momentum_" + str(round(K.get_value(getattr(optimizer, 'momentum')), 6))

    weights_file_path += ".weights.{epoch:03d}.h5"
    weights_file_path_best =  weights_file_path + ".best"

    # write model to yaml
    # please refer to the test case in test_attention_model.py for an example of restoring model from yaml

    yaml_file += "_model.yaml"
    with open(yaml_file, 'w') as file_:
        try:
            yaml = model.to_yaml()
            file_.write(yaml)
        except:
            logger.warning("to_yaml failed")

    # set call backs
    patience = int(os.getenv("TRAINING_PATIENCE", 10))
    callbacks = [EarlyStopping(patience=patience)]
    callbacks.append(ModelCheckpoint(filepath=weights_file_path_best, save_best_only=True, save_weights_only=True))

    if save_weights_for_each_epoch:
        callbacks.append(ModelCheckpoint(filepath=weights_file_path, save_weights_only=True))

    logger.info("Begin training")
    try:
        train_history = model.fit_generator(
            datasets["train"][1],
            samples_per_epoch=datasets["train"][0],
            nb_epoch=nb_epoch, verbose=1,
            callbacks=callbacks,
            validation_data=datasets["valid"][1],
            nb_val_samples=datasets["valid"][0],
            class_weight=None,
            max_q_size=1)

        # restore weights corresponding to the lowest val_loss
        val_loss = train_history.history['val_loss']
        best_epoch = np.argmin(val_loss)
        model.load_weights(weights_file_path_best.format(epoch=best_epoch))

    except MemoryError as e:
        logger.error("memory error!")
        logger.error("Mini batch too large for memory: %s" %
                       str(current_row.shape))
        raise

    logger.info("Training done")

    logger.info("Evaluating...")
    # evaluate
    evaluation_results = model.evaluate_generator( datasets["test"][1], val_samples=datasets["test"][0], max_q_size=1)
    print("evaluation results:")
    print(evaluation_results)
    logger.info("Evaluation done")


def infinit_generator(build_generator, mask_value):
    generator = build_generator()
    i = 0
    while True:
        try:
            n = next(generator)
            i += 1
            yield get_sequence_inputs(n[0],mask_value=mask_value), np.array(n[1])

        except StopIteration:
            if i == 0:
                raise StopIteration("build_generator builds an empty generator")

            i = 0
            generator = build_generator()

def limit_to(limit, generator):
    for i, e in enumerate(generator):
        if i >= limit:
            break

        yield e

def get_epoch_iterator(data_stream, n):
    global current_row
    data_stream.reset()
    for row in limit_to(n, data_stream.get_epoch_iterator()):
        current_row = row
        yield row


def load_datasets(aggregation_files, min_word_count, batch_size,
                  reduce_length_ratio_over_k_batches=12,
                  unpack=False, debug=False, mask_value=-1):
    datasets = dict()
    for partition, source in zip(partition_names, partition_sources):

        dataset = IMDBDatasetWrapper(
            source, min_word_count=min_word_count, encoding="utf-8",
            aggregation_files=aggregation_files, unpack=unpack)

        vocabulary = dataset.get_constrained_vocabulary()

        data_stream = dataset.get_indexable_data_stream()

        maximum_recursive_max_size =int(os.getenv("IMDB_MAXIMUM_RECURSIVE_MAX_SIZE", 42000))

        data_stream = dataset.mini_batch_stream_by_recursive_max_size(
            data_stream,
            batch_size=batch_size,
            maximum_recursive_max_size=  maximum_recursive_max_size  if partition == "train" else maximum_recursive_max_size * 5,
            reduce_length_ratio_over_k_batches=reduce_length_ratio_over_k_batches
            )

        data_stream = FilterSources(data_stream, data_stream.sources)

        n = dataset.n_samples

        iterator = infinit_generator(functools.partial(
            get_epoch_iterator, data_stream=data_stream, n=n),
            mask_value=mask_value)

        datasets[partition] = (n, iterator)

    return datasets, vocabulary
