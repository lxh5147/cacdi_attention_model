'''
Created on Jul 13, 2016

@author: lxh5147
'''
import os
import logging

if "THEANO_FLAGS" not in os.environ:
    os.environ["THEANO_FLAGS"] = "floatX=float32"
if "cnmem" not in os.environ["THEANO_FLAGS"]:
    os.environ["THEANO_FLAGS"] += ",lib.cnmem=1.0"

import theano

import numpy as np
random_seed = int( os.getenv('RANDOM_SEED',1321))
np.random.seed(random_seed)

from optimizers import SGDEx, RMSpropEx,AdadeltaEx

from attention_cacdi_exp_with_fuel_and_embedding_han_base import (
    cacdi_exp, load_datasets, logger, pretrained_word_embeddings_path)

logger = logging.getLogger()


def main(optimizer,
         hierarchical_layer_dropout_W=0,
         hierarchical_layer_dropout_U=0,
         attention_dropout=0.,
         attention_input_dropout=0.,
         mlp_softmax_classifier_input_drop_out=0,
         mlp_classifier_norm_inner_output=False,
         attention_norm_inner_output=False,
         save_weights_for_each_epoch=True,
         weights_file_path_prefix='',
         use_cnn_as_sequence_to_sequence_encoder=False,
         pooling_mode=None,
         initial_model_weights_file_path=None,
         debug=False,
         batch_size=32,
         reduce_length_ratio_over_k_batches=1,
         use_f1_to_early_stop=True,
         maximum_recursive_max_size=500000,
         maximum_recursive_max_size_for_dev = 5000000,
         samples_per_epoch = 0,
         word_embedding_dim = 50,
         use_per_class_threshold_tuning = True,
         evaluation_re_try_times=2,
         evaluation_re_try_waiting_time=0,
         fail_on_evlauation_failure=False,
         weight_regularizer_batch_norm = None,
         weight_regularizer_hidden = None,
         weight_regularizer_proj=None,
         weight_regularizer_encoder = None,
         weight_regularizer_attention = None,
         weight_regularizer_mlp_output = None,
         working_directory ='.',
         aggregation_files = ["sections.csv", "snapshots.csv"],
         unpack=False,
         output_dim = 50,
         input_window_size = 3,
         update_embedding = True,
         training_patience = 10,
         output_dims = None,
         input_window_sizes = None,
         ignore_classifier_weights = False,
         proj_learning_rate = None,
         classifier_hidden_unit_numbers = [],
         hidden_unit_activation_functions = [],
         nb_epoch = 100,
         initial_f1_on_dev = 0.,
         initial_threshold = None,
):

    datasets, vocabulary,labels_vocabulary  = load_datasets(
        aggregation_files=aggregation_files,
        min_word_count=5, unpack=unpack,
        batch_size=batch_size,
        reduce_length_ratio_over_k_batches=reduce_length_ratio_over_k_batches,
        debug=debug,
        maximum_recursive_max_size = maximum_recursive_max_size,
        maximum_recursive_max_size_for_dev = maximum_recursive_max_size_for_dev)

    #now dump the ids of train, dev and test for this dataset
    partition_names = ["train", "dev", "test"]
    for partition in partition_names:
        logger.info("export ids for %s" % partition)
        data_generator_with_id = datasets["cacdi"][partition][1]
        f = open(partition,'w')
        max = datasets["cacdi"][partition][0]
        count = 0
        visited_ids = set()
        should_break = False
        # this is a infinite generator
        for i in data_generator_with_id:
            ids = i[-1]
            for id in ids:
                if id in visited_ids:
                    should_break = True
                    break
                visited_ids.add(id)
                f.write(id)
                f.write('\n')
                count += 1
                if count >= max:
                    should_break = True
                    break
            if should_break:
                break
        logger.info("Total %d, actually wrote %d" %(max, count))
        f.close()

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--optimizer",
                        choices=['sgd','rmsprop','adadelta'],
                        default='sgd',
                        help="optimizer of the model, default sgd")

    parser.add_argument('--nb_epoch',
                        type=int,
                        default=100,
                        help="max number of training epoches")

    parser.add_argument("--base_learning_rate",
                        type=float,
                        default=0.001,
                        help="learning rate")

    parser.add_argument("--proj_learning_rate",
                        type=float,
                        default=0.001,
                        help="embedding learning rate")

    parser.add_argument("-c",
                        "--decay",
                        type=float,
                        default=0.,
                        help="learning rate decay")

    parser.add_argument("-t",
                        "--momentum",
                        type=float,
                        default=0.9,
                        help="momentum")

    parser.add_argument('--clipnorm',
                        type=float,
                        help="will normalize the gradient to the provided value")

    parser.add_argument("--mlp_softmax_classifier_input_drop_out",
                        type=float,
                        default=0,
                        help="drop out rate of mlp classifier input")

    parser.add_argument("--hierarchical_layer_dropout_W",
                        type=float,
                        default=0,
                        help="dropout on input of the hierarchical layer, or dropout of input of CNN input")

    parser.add_argument("--hierarchical_layer_dropout_U",
                        type=float,
                        default=0,
                        help="dropout on the recurrent layer - not applied on cnn")

    parser.add_argument("--attention_dropout",
                        type=float,
                        default=0,
                        help="dropout on the attention network")

    parser.add_argument("--attention_input_dropout",
                        type=float,
                        default=0,
                        help="dropout on the input of the attention network")

    parser.add_argument('--mlp_classifier_norm_inner_output',
                        action='store_true',
                        help="apply norm to the outputs of inner dense layers (not including the output layer) of mlp classifier")

    parser.add_argument('--attention_norm_inner_output',
                        action='store_true',
                        help="apply norm to the outputs of the hierarchical attention inner layers")

    parser.add_argument('--output_dim',
                        type=int,
                        default=75,
                        help="output dim of internal layers")

    parser.add_argument('--output_dims',
                        default='',
                        help="output dims of different layers from bottom to the top")

    parser.add_argument('--input_window_size',
                        type=int,
                        default=3,
                        help="CNN window size")

    parser.add_argument('--input_window_sizes',
                        default='',
                        help="CNN window sizes for different CNN layers, from bottom to the top")

    parser.add_argument('--maximum_recursive_max_size',
                        type=int,
                        default=500000,
                        help="max size of words in a mini batch including padding for training dataset")

    parser.add_argument('--maximum_recursive_max_size_for_dev',
                        type=int,
                        default=5000000,
                        help="max size of words in a mini batch for prediction including padding for dev dataset")

    parser.add_argument("--weight_regularizer_batch_norm",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_batch_norm",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_batch_norm",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")

    parser.add_argument("--weight_regularizer_hidden",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_hidden",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_hidden",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")

    parser.add_argument("--weight_regularizer_proj",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_proj",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_proj",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")

    parser.add_argument("--weight_regularizer_encoder",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_encoder",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_encoder",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")


    parser.add_argument("--weight_regularizer_attention",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_attention",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_attention",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")

    parser.add_argument("--weight_regularizer_mlp_output",
                        choices=['none','l1l2', 'l1', 'l2'],
                        default='none',
                        help="weight regularization method")

    parser.add_argument("--l1_regularizer_weight_mlp_output",
                        type=float,
                        default=1e-7,
                        help="weight of l1 weight regularizer")

    parser.add_argument("--l2_regularizer_weight_mlp_output",
                        type=float,
                        default=1e-7,
                        help="weight of l2 weight regularizer")

    parser.add_argument('--samples_per_epoch',
                        type=int,
                        default=0,
                        help="samples per epoch")

    parser.add_argument('--training_patience',
                        type=int,
                        default=10,
                        help="patience of training-- stop training if no improvements after this number of epochs of training")

    parser.add_argument('--word_embedding_dim',
                        type=int,
                        default=50,
                        help="word embedding dim")

    parser.add_argument('--save_weights_for_each_epoch',
                        action='store_true',
                        help="save weights for each epoch")

    parser.add_argument('--weights_file_path_prefix',
                        default='',
                        help="prefix of weight and model file, to run multiple experiments under the same folder")

    parser.add_argument('--use_cnn_as_sequence_to_sequence_encoder',
                        action='store_true',
                        help="use cnn as sequence_to_sequence encoder")

    parser.add_argument("--pooling_mode",
                        choices=['avg', 'max','none'],
                        default='none',
                        help="pooling mode,default is none")

    parser.add_argument('--batch',
                        type=int,
                        default=32,
                        help="batch size - default = 32")

    parser.add_argument('--reduce-length-ratio-over-k-batches',
                        type=int,
                        default=4,
                        help="reduces the length ratio over k batches")

    parser.add_argument('--initial_model_weights_file_path',
                        default='',
                        help="model weights file path, generated by a previous "
                             "training. If this parameter is set, current "
                             "training will fine tune previous training, "
                             "instead of training a new model from scratch.")

    parser.add_argument('--initial_f1_on_dev',
                        default=0.,
                        type =float,
                        help="initial f1 on dev dataset")

    parser.add_argument('--initial_threshold',
                        default='',
                        help="initial threshold file or threshold")

    parser.add_argument('--ignore_classifier_weights',
                        action='store_true',
                        help="ignore weights of the output classifier when loading weights from a weight file")

    parser.add_argument('--use_f1_to_early_stop',
                        action='store_true',
                        help="use external script to calculate the f1, and use this acc to early stop")

    parser.add_argument('--use_per_class_threshold_tuning',
                        action='store_true',
                        help="use per class threshold for tuning")

    parser.add_argument('--evaluation_re_try_times',
                        type=int,
                        default=2,
                        help="re-try times in case of failure of evaluation")

    parser.add_argument("--evaluation_re_try_waiting_time",
                        type=float,
                        default=0.,
                        help="waiting time between evaluation re-try")

    parser.add_argument('--fail_on_evlauation_failure',
                        action='store_true',
                        help="fail the whole exp if an evaluation fails")

    parser.add_argument("--working_directory",
                        default='.',
                        help="working directory of current experiment, into which temporal stuff will be written")

    parser.add_argument("--aggregation_files",
                        default='sections.csv,snapshots.csv',
                        help="aggregation files to be used by the CACDI dataset")

    parser.add_argument('--unpack',
                        action='store_true',
                        help="unpack a sentence into words, i.e., removing sentence layer")

    parser.add_argument('--not_update_embedding',
                        action='store_true',
                        help="not update embeddings during training")

    parser.add_argument("--classifier_hidden_unit_numbers",
                        default='',
                        help="classifier hidden unit numbers,separated by ',', e.g., 50,50")

    parser.add_argument("--hidden_unit_activation_functions",
                        default='',
                        help="hidden unit activation functions, separated by ',', e.g., relu,relu")


    args = parser.parse_args()

    learning_rate = args.base_learning_rate
    decay = args.decay
    momentum = args.momentum

    if args.optimizer == 'sgd':
        logger.info('using SGD optimizer')
        if args.clipnorm is not None:
            logger.info('using clipnorm {}'.format(args.clipnorm))
            optimizer = SGDEx(lr=learning_rate, decay=decay, momentum=momentum,
                            clipnorm=args.clipnorm)
        else:
            optimizer = SGDEx(lr=learning_rate, decay=decay, momentum=momentum)

    elif args.optimizer == 'rmsprop':
        logger.info('using RMSprop optimizer')
        if args.clipnorm is not None:
            logger.info('using clipnorm {}'.format(args.clipnorm))
            optimizer = RMSpropEx(lr=learning_rate, decay=decay,
                                clipnorm=args.clipnorm)
        else:
            optimizer = RMSpropEx(lr=learning_rate,decay=decay)

    else:
        logger.info('using Adadelta optimizer')
        if args.clipnorm is not None:
            logger.info('using clipnorm {}'.format(args.clipnorm))
            optimizer = AdadeltaEx(lr=learning_rate, decay=decay,
                                clipnorm=args.clipnorm)
        else:
            optimizer = AdadeltaEx(lr=learning_rate,decay=decay)

    logger.info('base learning rate: {}'.format(args.base_learning_rate))
    logger.info('proj learning rate: {}'.format(args.proj_learning_rate))

    logger.info('momentum: {}'.format(args.momentum))
    logger.info('decay: {}'.format(args.decay))
    logger.info('word_embedding_dim: {}'.format(args.word_embedding_dim))

    logger.info('dropout levels: mlp softmax {}, hierarchical_layer_dropout_W '
                '{}, hierarchical_layer_dropout_U {}, '
                ' attention_dropout {},'
                ' attention_input_dropout {}'
                ''.format(
        args.mlp_softmax_classifier_input_drop_out,
        args.hierarchical_layer_dropout_W, args.hierarchical_layer_dropout_U,
        args.attention_dropout,
        args.attention_input_dropout))

    logger.info('seed is {}'.format(random_seed))
    logger.info('python executable coming from {}'.format(sys.executable))
    try:
        logger.info('is theano deterministic? {}'.format(
            theano.config.deterministic))
    except AttributeError:
        logger.info('theano is NOT deterministic')

    logger.info('maximum_recursive_max_size:%s' % args.maximum_recursive_max_size)
    logger.info('maximum_recursive_max_size_for_dev:%s' % args.maximum_recursive_max_size_for_dev)

    logger.info('samples_per_epoch:%s' % args.samples_per_epoch)

    logger.info('use_per_class_threshold_tuning:%s' % args.use_per_class_threshold_tuning)

    logger.info('evaluation_re_try_times:%s' % args.evaluation_re_try_times)
    logger.info('evaluation_re_try_waiting_time:%s' % args.evaluation_re_try_waiting_time)
    logger.info('fail_on_evlauation_failure:%s' % args.fail_on_evlauation_failure)

    if args.weight_regularizer_hidden == 'none':
        logger.info('weight_regularizer_hidden:No')
        weight_regularizer_hidden = None
    elif args.weight_regularizer_hidden == 'l1l2':
        logger.info('weight_regularizer_hidden:%s, l1_regularizer_weight_hidden:%s,l2_regularizer_weight_hidden:%s' % (
        args.weight_regularizer_hidden, args.l1_regularizer_weight_hidden, args.l2_regularizer_weight_hidden))
        from keras.regularizers import l1l2
        weight_regularizer_hidden = l1l2(l1=args.l1_regularizer_weight_hidden, l2=args.l2_regularizer_weight_hidden)
    elif args.weight_regularizer_hidden == 'l1':
        logger.info('weight_regularizer_hidden:%s, l1_regularizer_weight_hidden:%s' % (
        args.weight_regularizer_hidden, args.l1_regularizer_weight_hidden))
        from keras.regularizers import l1
        weight_regularizer_hidden = l1(l=args.l1_regularizer_weight_hidden)
    elif args.weight_regularizer_hidden == 'l2':
        logger.info('weight_regularizer_hidden:%s, l2_regularizer_weight_hidden:%s' % (
        args.weight_regularizer_hidden, args.l2_regularizer_weight_hidden))
        from keras.regularizers import l2
        weight_regularizer_hidden = l2(l=args.l2_regularizer_weight_hidden)

    if weight_regularizer_hidden:
        weight_regularizer_hidden = weight_regularizer_hidden.get_config()

    #weight_regularizer_proj
    if args.weight_regularizer_proj == 'none':
        logger.info('weight_regularizer_proj:No')
        weight_regularizer_proj = None
    elif args.weight_regularizer_proj == 'l1l2':
        logger.info('weight_regularizer_proj:%s, l1_regularizer_weight_proj:%s,l2_regularizer_weight_proj:%s' % (
        args.weight_regularizer_proj, args.l1_regularizer_weight_proj, args.l2_regularizer_weight_proj))
        from keras.regularizers import l1l2
        weight_regularizer_proj = l1l2(l1=args.l1_regularizer_weight_proj, l2=args.l2_regularizer_weight_proj)
    elif args.weight_regularizer_proj == 'l1':
        logger.info('weight_regularizer_proj:%s, l1_regularizer_weight_proj:%s' % (
        args.weight_regularizer_proj, args.l1_regularizer_weight_proj))
        from keras.regularizers import l1
        weight_regularizer_proj = l1(l=args.l1_regularizer_weight_proj)
    elif args.weight_regularizer_proj == 'l2':
        logger.info('weight_regularizer_proj:%s, l2_regularizer_weight:%s' % (
        args.weight_regularizer_proj, args.l2_regularizer_weight_proj))
        from keras.regularizers import l2
        weight_regularizer_proj = l2(l=args.l2_regularizer_weight_proj)

    if weight_regularizer_proj:
        weight_regularizer_proj = weight_regularizer_proj.get_config()

    #weight_regularizer_encoder
    if args.weight_regularizer_encoder == 'none':
        logger.info('weight_regularizer_encoder:No')
        weight_regularizer_encoder = None
    elif args.weight_regularizer_encoder == 'l1l2':
        logger.info('weight_regularizer_encoder:%s, l1_regularizer_weight_encoder:%s,l2_regularizer_weight_encoder:%s' % (
        args.weight_regularizer_encoder, args.l1_regularizer_weight_encoder, args.l2_regularizer_weight_encoder))
        from keras.regularizers import l1l2
        weight_regularizer_encoder = l1l2(l1=args.l1_regularizer_weight_encoder, l2=args.l2_regularizer_weight_encoder)
    elif args.weight_regularizer_encoder == 'l1':
        logger.info('weight_regularizer_encoder:%s, l1_regularizer_weight_encoder:%s' % (
        args.weight_regularizer_encoder, args.l1_regularizer_weight_encoder))
        from keras.regularizers import l1
        weight_regularizer_encoder = l1(l=args.l1_regularizer_weight_encoder)
    elif args.weight_regularizer_encoder == 'l2':
        logger.info('weight_regularizer_encoder:%s, l2_regularizer_weight:%s' % (
        args.weight_regularizer_encoder, args.l2_regularizer_weight_encoder))
        from keras.regularizers import l2
        weight_regularizer_encoder = l2(l=args.l2_regularizer_weight_encoder)

    if weight_regularizer_encoder:
        weight_regularizer_encoder = weight_regularizer_encoder.get_config()


    #weight_regularizer_attention
    if args.weight_regularizer_attention == 'none':
        logger.info('weight_regularizer_attention:No')
        weight_regularizer_attention = None
    elif args.weight_regularizer_attention == 'l1l2':
        logger.info('weight_regularizer_attention:%s, l1_regularizer_weight_attention:%s,l2_regularizer_weight_attention:%s' % (
        args.weight_regularizer_attention, args.l1_regularizer_weight_attention, args.l2_regularizer_weight_attention))
        from keras.regularizers import l1l2
        weight_regularizer_attention = l1l2(l1=args.l1_regularizer_weight_attention, l2=args.l2_regularizer_weight_attention)
    elif args.weight_regularizer_attention == 'l1':
        logger.info('weight_regularizer_attention:%s, l1_regularizer_weight_attention:%s' % (
        args.weight_regularizer_attention, args.l1_regularizer_weight_attention))
        from keras.regularizers import l1
        weight_regularizer_attention = l1(l=args.l1_regularizer_weight_attention)
    elif args.weight_regularizer_attention == 'l2':
        logger.info('weight_regularizer_attention:%s, l2_regularizer_weight:%s' % (
        args.weight_regularizer_attention, args.l2_regularizer_weight_attention))
        from keras.regularizers import l2
        weight_regularizer_attention = l2(l=args.l2_regularizer_weight_attention)

    if weight_regularizer_attention:
        weight_regularizer_attention = weight_regularizer_attention.get_config()

    #weight_regularizer_mlp_output
    if args.weight_regularizer_mlp_output == 'none':
        logger.info('weight_regularizer_mlp_output:No')
        weight_regularizer_mlp_output = None
    elif args.weight_regularizer_mlp_output == 'l1l2':
        logger.info('weight_regularizer_mlp_output:%s, l1_regularizer_weight_mlp_output:%s,l2_regularizer_weight_mlp_output:%s' % (
        args.weight_regularizer_mlp_output, args.l1_regularizer_weight_mlp_output, args.l2_regularizer_weight_mlp_output))
        from keras.regularizers import l1l2
        weight_regularizer_mlp_output = l1l2(l1=args.l1_regularizer_weight_mlp_output, l2=args.l2_regularizer_weight_mlp_output)
    elif args.weight_regularizer_mlp_output == 'l1':
        logger.info('weight_regularizer_mlp_output:%s, l1_regularizer_weight_mlp_output:%s' % (
        args.weight_regularizer_mlp_output, args.l1_regularizer_weight_mlp_output))
        from keras.regularizers import l1
        weight_regularizer_mlp_output = l1(l=args.l1_regularizer_weight_mlp_output)
    elif args.weight_regularizer_mlp_output == 'l2':
        logger.info('weight_regularizer_mlp_output:%s, l2_regularizer_weight:%s' % (
        args.weight_regularizer_mlp_output, args.l2_regularizer_weight_mlp_output))
        from keras.regularizers import l2
        weight_regularizer_mlp_output = l2(l=args.l2_regularizer_weight_mlp_output)

    if weight_regularizer_mlp_output:
        weight_regularizer_mlp_output = weight_regularizer_mlp_output.get_config()


    #weight_regularizer_batch_norm
    if args.weight_regularizer_batch_norm == 'none':
        logger.info('weight_regularizer_batch_norm:No')
        weight_regularizer_batch_norm = None
    elif args.weight_regularizer_batch_norm == 'l1l2':
        logger.info('weight_regularizer_batch_norm:%s, l1_regularizer_weight_batch_norm:%s,l2_regularizer_weight_batch_norm:%s' % (
        args.weight_regularizer_batch_norm, args.l1_regularizer_weight_batch_norm, args.l2_regularizer_weight_batch_norm))
        from keras.regularizers import l1l2
        weight_regularizer_batch_norm = l1l2(l1=args.l1_regularizer_weight_batch_norm, l2=args.l2_regularizer_weight_batch_norm)
    elif args.weight_regularizer_batch_norm == 'l1':
        logger.info('weight_regularizer_batch_norm:%s, l1_regularizer_weight_batch_norm:%s' % (
        args.weight_regularizer_batch_norm, args.l1_regularizer_weight_batch_norm))
        from keras.regularizers import l1
        weight_regularizer_batch_norm = l1(l=args.l1_regularizer_weight_batch_norm)
    elif args.weight_regularizer_batch_norm == 'l2':
        logger.info('weight_regularizer_batch_norm:%s, l2_regularizer_weight:%s' % (
        args.weight_regularizer_batch_norm, args.l2_regularizer_weight_batch_norm))
        from keras.regularizers import l2
        weight_regularizer_batch_norm = l2(l=args.l2_regularizer_weight_batch_norm)

    if weight_regularizer_batch_norm:
        weight_regularizer_batch_norm = weight_regularizer_batch_norm.get_config()


    logger.info('working_directory:%s' % args.working_directory)
    if not os.path.exists(args.working_directory):
        os.makedirs(args.working_directory)
    assert os.path.isdir(args.working_directory)

    logger.info('aggregation_files:%s' % args.aggregation_files)

    logger.info('unpack:%s' % args.unpack)

    logger.info('output_dim:%s' % args.output_dim)

    logger.info('input_window_size:%s' % args.input_window_size)

    logger.info('not_update_embedding:%s' %  args.not_update_embedding)

    logger.info('training_patience:%s' % args.training_patience)

    logger.info('output_dims:%s' % args.output_dims)

    logger.info('input_window_sizes:%s' % args.input_window_sizes)

    logger.info('ignore_classifier_weights:%s' % args.ignore_classifier_weights)

    logger.info('classifier_hidden_unit_numbers:%s' % args.classifier_hidden_unit_numbers)
    logger.info('hidden_unit_activation_functions:%s' % args.hidden_unit_activation_functions)

    # from bottom to top
    classifier_hidden_unit_numbers = []
    if args.classifier_hidden_unit_numbers:
        classifier_hidden_unit_numbers = [int(i) for i in args.classifier_hidden_unit_numbers.split(',')]

    hidden_unit_activation_functions = []
    if args.hidden_unit_activation_functions:
        hidden_unit_activation_functions = args.hidden_unit_activation_functions.split(',')

    logger.info('nb_epoch:%s' % args.nb_epoch)

    logger.info('initial_f1_on_dev:%s' % args.initial_f1_on_dev)
    logger.info('initial_threshold:%s' % args.initial_threshold)

    # some check if provided initial model weights
    initial_threshold = None
    if args.initial_model_weights_file_path:
        assert args.initial_f1_on_dev >=0
        assert args.initial_threshold
        if args.use_per_class_threshold_tuning:
            # threshold is a file
            assert os.path.exists(args.initial_threshold)
            initial_threshold = args.initial_threshold
        else:
            initial_threshold = float(args.initial_threshold)

    main(optimizer,
         mlp_softmax_classifier_input_drop_out=args.mlp_softmax_classifier_input_drop_out,
         hierarchical_layer_dropout_W=args.hierarchical_layer_dropout_W,
         hierarchical_layer_dropout_U=args.hierarchical_layer_dropout_U,
         attention_dropout=args.attention_dropout,
         attention_input_dropout=args.attention_input_dropout,
         mlp_classifier_norm_inner_output = args.mlp_classifier_norm_inner_output,
         save_weights_for_each_epoch = args.save_weights_for_each_epoch,
         attention_norm_inner_output = args.attention_norm_inner_output,
         weights_file_path_prefix = args.weights_file_path_prefix,
         use_cnn_as_sequence_to_sequence_encoder = args.use_cnn_as_sequence_to_sequence_encoder,
         pooling_mode = None if args.pooling_mode == 'none' else args.pooling_mode,
         use_f1_to_early_stop = args.use_f1_to_early_stop,
         initial_model_weights_file_path = args.initial_model_weights_file_path,
         debug="--debug" in sys.argv, batch_size=args.batch,
         reduce_length_ratio_over_k_batches=args.reduce_length_ratio_over_k_batches,
         maximum_recursive_max_size=args.maximum_recursive_max_size,
         samples_per_epoch=args.samples_per_epoch,
         word_embedding_dim = args.word_embedding_dim,
         use_per_class_threshold_tuning=args.use_per_class_threshold_tuning,
         evaluation_re_try_times=args.evaluation_re_try_times,
         evaluation_re_try_waiting_time=args.evaluation_re_try_waiting_time,
         fail_on_evlauation_failure=args.fail_on_evlauation_failure,
         weight_regularizer_batch_norm = weight_regularizer_batch_norm,
         weight_regularizer_hidden=weight_regularizer_hidden,
         weight_regularizer_proj=weight_regularizer_proj,
         weight_regularizer_encoder = weight_regularizer_encoder,
         weight_regularizer_attention = weight_regularizer_attention,
         weight_regularizer_mlp_output =weight_regularizer_mlp_output,
         working_directory=args.working_directory,
         aggregation_files=args.aggregation_files.split(','),
         unpack=args.unpack,
         output_dim=args.output_dim,
         input_window_size=args.input_window_size,
         maximum_recursive_max_size_for_dev = args.maximum_recursive_max_size_for_dev,
         update_embedding = not args.not_update_embedding,
         training_patience = args.training_patience,
         output_dims = args.output_dims,
         input_window_sizes = args.input_window_sizes,
         ignore_classifier_weights = args.ignore_classifier_weights,
         proj_learning_rate = args.proj_learning_rate,
         classifier_hidden_unit_numbers = classifier_hidden_unit_numbers,
         hidden_unit_activation_functions = hidden_unit_activation_functions,
         nb_epoch = args.nb_epoch,
         initial_f1_on_dev = args.initial_f1_on_dev,
         initial_threshold = initial_threshold,
         )

