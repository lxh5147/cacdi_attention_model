'''
Created on Jul 13, 2016

@author: lxh5147
'''
import os

if "THEANO_FLAGS" not in os.environ:
    os.environ["THEANO_FLAGS"] = "floatX=float32"
if "cnmem" not in os.environ["THEANO_FLAGS"]:
    os.environ["THEANO_FLAGS"] += ",lib.cnmem=1.0"

import numpy as np

random_seed = int(os.getenv('RANDOM_SEED', 1321))
np.random.seed(random_seed)

from keras.optimizers import SGD, RMSprop, Adagrad, Adam

from attention_imdb_exp_with_fuel_base import (
    imdb_exp, load_datasets, logger, pretrained_word_embeddings_path)

import theano


def main(optimizer,
         hierarchical_layer_dropout_W=0,
         hierarchical_layer_dropout_U=0,
         attention_context_dropout=0.,
         attention_dropout=0.,
         attention_input_dropout=0.,
         mlp_softmax_classifier_input_drop_out=0,
         mlp_classifier_norm_inner_output=False,
         attention_norm_inner_output=False,
         save_weights_for_each_epoch=True,
         apply_input_mask=False,
         weights_file_path_prefix='',
         use_cnn_as_sequence_to_sequence_encoder=False,
         pooling_mode=None,
         unpack=False,  # if unpacked, flat level
         initial_model_weights_file_path=None,
         debug=False, batch_size=32,
         reduce_length_ratio_over_k_batches=1):
    if debug:
        aggregation_files = ["curated_comments_debug.csv"]
    else:
        aggregation_files = ["curated_comments.csv"]

    logger.info('batch size {}'.format(batch_size))
    logger.info('reduce_length_ratio_over_k_batches size {}'.format(
        reduce_length_ratio_over_k_batches))

    datasets, vocabulary = load_datasets(
        aggregation_files=aggregation_files,
        min_word_count=5,
        unpack=unpack,
        batch_size=batch_size,
        reduce_length_ratio_over_k_batches=reduce_length_ratio_over_k_batches,
        debug=debug)

    logger.info("mask value i.e., id of </s> for this vacabulary is %s" % vocabulary["</s>"])

    max_sentences = None
    max_words = None

    sentence_output_dim = 50
    word_output_dim = 50

    sentence_attention_weight_vec_dim = 100
    word_attention_weight_vec_dim = 100

    vocabulary_size = len(vocabulary)  # 100
    word_embedding_dim = 200  # 200
    logger.info("loading pre-trained word embeddings: %s" %
                pretrained_word_embeddings_path)
    initial_embedding = np.load(pretrained_word_embeddings_path)
    if initial_embedding.shape[0] != vocabulary_size:
        raise ValueError("Word embeddings have bad vocabulary size: %d but "
                         "should be %d" %
                         (initial_embedding.shape[0], vocabulary_size))

    if initial_embedding.shape[1] != word_embedding_dim:
        raise ValueError("Word embeddings have bad vector size: %d but "
                         "should be %d" %
                         (initial_embedding.shape[1], word_embedding_dim))
    classifier_output_dim = 10
    classifier_hidden_unit_numbers = []
    hidden_unit_activation_functions = []

    # when use cnn as sequence, we must set the input window size
    if use_cnn_as_sequence_to_sequence_encoder:
        if unpack:
            # word -> comment
            input_window_sizes = (3,)
        else:
            # word -> sentence -> comment
            input_window_sizes = (3, 3)
    else:
        input_window_sizes = None

    imdb_exp(
        datasets,
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
        use_cnn_as_sequence_to_sequence_encoder=use_cnn_as_sequence_to_sequence_encoder,
        optimizer=optimizer,
        pooling_mode=pooling_mode,
        save_weights_for_each_epoch=save_weights_for_each_epoch,
        mlp_softmax_classifier_input_drop_out=mlp_softmax_classifier_input_drop_out,
        hierarchical_layer_dropout_W=hierarchical_layer_dropout_W,
        hierarchical_layer_dropout_U=hierarchical_layer_dropout_U,
        attention_context_dropout=attention_context_dropout,
        attention_dropout=attention_dropout,
        attention_input_dropout=attention_input_dropout,
        mlp_classifier_norm_inner_output=mlp_classifier_norm_inner_output,
        attention_norm_inner_output=attention_norm_inner_output,
        input_mask_value=vocabulary["</s>"],
        apply_input_mask=apply_input_mask,
        weights_file_path_prefix=weights_file_path_prefix,
        unpacked=unpack,
        input_window_sizes=input_window_sizes,
        initial_model_weights_file_path=initial_model_weights_file_path)


if __name__ == '__main__':
    # http://aclweb.org/anthology/N16-1174
    import sys
    import argparse

    # support optimizer and learning rate for sgd
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--optimizer",
                        choices=['sgd', 'rmsprop', 'adagrad', 'adam'],
                        default='sgd',
                        help="optimizer of the model, default sgd")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-c", "--decay", type=float, default=0., help="learning rate decay")
    parser.add_argument("-t", "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("-d", "--mlp_softmax_classifier_input_drop_out",
                        type=float, default=0,
                        help="drop out rate of mlp classifier input")
    parser.add_argument("--hierarchical_layer_dropout_W",
                        type=float, default=0,
                        help="dropout on input of the hierarchical layer")
    parser.add_argument("--hierarchical_layer_dropout_U",
                        type=float, default=0,
                        help="dropout on the recurrent layer - not applied on cnn")

    parser.add_argument("--attention_context_dropout",
                        type=float,
                        help="dropout on the context of attention network")
    parser.add_argument("--attention_dropout",
                        type=float,
                        help="dropout on the attention network")
    parser.add_argument("--attention_input_dropout",
                        type=float,
                        help="dropout on the input of the attention network")
    parser.add_argument('--save_weights_for_each_epoch', action='store_true', help="save weights for each epoch")
    parser.add_argument('--mlp_classifier_norm_inner_output', action='store_true',
                        help="apply norm to the outputs of inner dense layers (not including the output layer) of mlp classifier")
    parser.add_argument('--attention_norm_inner_output', action='store_true',
                        help="apply norm to the outputs of the hierarchical attention inner layers")
    parser.add_argument('--apply_input_mask', action='store_true', help="apply mask to the model")
    parser.add_argument('--weights_file_path_prefix', default='',
                        help="prefix of weight and model file, to run multiple experiments under the same folder")
    parser.add_argument('--use_cnn_as_sequence_to_sequence_encoder', action='store_true',
                        help="use cnn as sequence_to_sequence encoder")
    parser.add_argument("--pooling_mode", choices=['avg', 'max', 'none'], default='none',
                        help="pooling mode,default is none")
    parser.add_argument('--unpack', action='store_true', help="unpack a sentence into words, for IMDB, no hierarchy")
    parser.add_argument('--batch', type=int, default=32, help="batch size - default = 32")
    parser.add_argument('--reduce-length-ratio-over-k-batches', type=int,
                        default=4,
                        help="reduces the length ratio over k batches")
    parser.add_argument('--initial_model_weights_file_path', default='',
                        help="model weights file path, generated by a previous "
                             "training. If this parameter is set, current "
                             "training will fine tune previous training, "
                             "instead of training a new model from scratch.")
    parser.add_argument('--clipnorm', type=float,
                        help="will normalize the gradient to the provided "
                             "value")
    parser.add_argument('--clipvalue', type=float,
                        help="will clip the gradient to the provided "
                             "value")
    args = parser.parse_args()

    options = {'lr': args.learning_rate, 'decay': args.decay,
               'momentum': args.momentum}
    logger.info('learning rate: {}'.format(args.learning_rate))
    logger.info('momentum: {}'.format(args.momentum))
    logger.info('decay: {}'.format(args.decay))

    if args.clipvalue is not None:
        logger.info('using clipvalue {}'.format(args.clipvalue))
        options['clipvalue'] = args.clipvalue
    if args.clipnorm is not None:
        logger.info('using clipnorm {}'.format(args.clipnorm))
        options['clipnorm'] = args.clipnorm

    if args.optimizer == 'sgd':
        logger.info('using SGD optimizer')
        exp_optimizer = SGD
    elif args.optimizer == 'adagrad':
        logger.info('using Adagrad optimizer')
        options.pop('momentum', None)  # momentum option not accepted by Adagrad
        exp_optimizer = Adagrad
    elif args.optimizer == 'adam':
        logger.info('using Adam optimizer')
        options.pop('momentum', None)  # momentum option not accepted by Adam
        exp_optimizer = Adam
    else:
        assert 'rmsprop' == args.optimizer
        logger.info('using RMSprop optimizer')
        options.pop('momentum', None)  # momentum option not accepted by RMSProp
        exp_optimizer = RMSprop

    logger.info('optimizer params: {}'.format(options))

    optimizer = exp_optimizer(**options)

    if args.attention_context_dropout is None:
        args.attention_context_dropout = args.hierarchical_layer_dropout_W
    if args.attention_dropout is None:
        args.attention_dropout = args.hierarchical_layer_dropout_W
    if args.attention_input_dropout is None:
        args.attention_input_dropout = args.hierarchical_layer_dropout_W

    logger.info('dropout levels: mlp softmax {}, hierarchical_layer_dropout_W '
                '{}, hierarchical_layer_dropout_U {}, '
                'attention_context_dropout {}, attention_dropout {},'
                ' attention_input_dropout {}'
                ''.format(
        args.mlp_softmax_classifier_input_drop_out,
        args.hierarchical_layer_dropout_W, args.hierarchical_layer_dropout_U,
        args.attention_context_dropout, args.attention_dropout,
        args.attention_input_dropout))

    logger.info('seed is {}'.format(random_seed))
    logger.info('python executable coming from {}'.format(sys.executable))
    try:
        logger.info('is theano deterministic? {}'.format(
            theano.config.deterministic))
    except AttributeError:
        logger.info('theano is NOT deterministic')

    main(optimizer,
         mlp_softmax_classifier_input_drop_out=args.mlp_softmax_classifier_input_drop_out,
         hierarchical_layer_dropout_W=args.hierarchical_layer_dropout_W,
         hierarchical_layer_dropout_U=args.hierarchical_layer_dropout_U,
         attention_context_dropout=args.attention_context_dropout,
         attention_dropout=args.attention_dropout,
         attention_input_dropout=args.attention_input_dropout,
         mlp_classifier_norm_inner_output=args.mlp_classifier_norm_inner_output,
         save_weights_for_each_epoch=args.save_weights_for_each_epoch,
         attention_norm_inner_output=args.attention_norm_inner_output,
         apply_input_mask=args.apply_input_mask,
         weights_file_path_prefix=args.weights_file_path_prefix,
         use_cnn_as_sequence_to_sequence_encoder=args.use_cnn_as_sequence_to_sequence_encoder,
         pooling_mode=None if args.pooling_mode == 'none' else args.pooling_mode,
         unpack=args.unpack,
         initial_model_weights_file_path=args.initial_model_weights_file_path,
         debug="--debug" in sys.argv, batch_size=args.batch,
         reduce_length_ratio_over_k_batches=args.reduce_length_ratio_over_k_batches)
