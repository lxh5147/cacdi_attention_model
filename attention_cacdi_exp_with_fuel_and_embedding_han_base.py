'''
Created on Jul 13, 2016

@author: lxh5147
'''
from collections import defaultdict
import cPickle
import functools
import logging
import os
import keras.backend as K
import numpy as np
import time
np.random.seed(1321)


from fuel.transformers import Mapping, FilterSources

from fuel_cacdi.datasets.cacdi import CACDIDatasetWrapper
from fuel_cacdi.extra_converters import cacdi as cacdi_converter
from fuel_cacdi.transformers import find_maximum_padding_size, HierarchyLevel


from HierarchicalEmbeddingWithAttentionLayer import  build_classifier_with_hierarchical_embedding_attention
from tree_tensor_converter import  get_sequence_inputs

from callbacks_for_embedding_han import EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

source_dir = os.getenv('CACDI_SOURCE_DIR', '/hcnlp/users/xiaohua_liu/cacdi_data/singleton_complete')


train_source = os.path.join(source_dir, "train")
dev_source = os.path.join(source_dir, "dev")
test_source = os.path.join(source_dir, "test")

partition_names = ["train", "dev", "test"]
partition_sources = [train_source, dev_source, test_source]

pretrained_word_embeddings_path = os.path.join(
    source_dir,
    os.getenv('CACDI_WORD_EMBEDDING_FILE_NAME', "train_dev_set_w2v_50_5.npy"))

LABELS_INDEX = cacdi_converter.Snapshot.BASE_HEADER.index("labels")
SNAPSHOT_ID_INDEX = cacdi_converter.Snapshot.BASE_HEADER.index("snapshot id")
DOCUMENT_TYPE_INDEX = cacdi_converter.Document.BASE_HEADER.index("document type")
SECTION_DOCUMENT_TYPE_INDEX = cacdi_converter.Section.BASE_HEADER.index("document type")
SECTION_NAME_INDEX = cacdi_converter.Section.BASE_HEADER.index("section name")


current_row = None

# x_train, y_train, x_test, y_test, x_pred, batch_size, nb_epoch,
def cacdi_exp(datasets, 
              input_shape,
              output_dims,
              attention_weight_vector_dims,
              vocabulary_size, 
              word_embedding_dim, 
              initial_embedding, 
              classifier_output_dim, 
              classifier_hidden_unit_numbers, 
              hidden_unit_activation_functions,
              optimizer,
              use_cnn_as_sequence_to_sequence_encoder=False,
              input_window_sizes=None,
              pooling_mode=None,
              save_weights_for_each_epoch=True,
              mlp_softmax_classifier_input_drop_out=0.,
              hierarchical_layer_dropout_W=0.,
              hierarchical_layer_dropout_U=0.,
              attention_dropout=0.,
              attention_input_dropout=0.,
              mlp_classifier_norm_inner_output=False,
              attention_norm_inner_output=False,
              input_mask_value=-1,
              weights_file_path_prefix='',
              initial_model_weights_file_path=None,
              use_f1_to_early_stop=True,
              samples_per_epoch = 0,
              labels_vocabulary = None,
              use_per_class_threshold_tuning=True,
              on_train_on_batch_failed = None,
              evaluation_re_try_times=2,
              evaluation_re_try_waiting_time=0,
              fail_on_evlauation_failure=False,
              weight_regularizer_batch_norm = None,
              weight_regularizer_hidden = None,
              weight_regularizer_proj = None,
              weight_regularizer_encoder = None,
              weight_regularizer_attention = None,
              weight_regularizer_mlp_output = None,
              working_directory = '.',
              update_embedding = True,
              training_patience = 10,
              ignore_classifier_weights = False,
              proj_learning_rate = None,
              ):

    if initial_model_weights_file_path:
        #check if model weights file exist
        if not os.path.exists(initial_model_weights_file_path):
            print("Initial model weights file does not exist: %s" % initial_model_weights_file_path)
            return
        if not os.path.isfile(initial_model_weights_file_path):
            print("Initial model weights file is a directory, should be a file: %s" % initial_model_weights_file_path)
            return

    # TODO: support different level features
    input_feature_dims = (0, ) * len(input_shape)
    logger.info("Input feature dims=%s" % list(input_feature_dims))

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
        output_activation_function="sigmoid",
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
        input_mask_value = input_mask_value,
        on_train_on_batch_failed = on_train_on_batch_failed,
        weight_regularizer_batch_norm = weight_regularizer_batch_norm,
        weight_regularizer_hidden=weight_regularizer_hidden,
        weight_regularizer_proj = weight_regularizer_proj,
        weight_regularizer_encoder = weight_regularizer_encoder,
        weight_regularizer_attention = weight_regularizer_attention,
        weight_regularizer_mlp_output = weight_regularizer_mlp_output,
        update_embedding = update_embedding,
        proj_learning_rate = proj_learning_rate)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

    # pre-load weights, the caller should ensure that the weights file is valid for this model
    if initial_model_weights_file_path:
        logger.info("Initialize model weights by loading weights from %s" % initial_model_weights_file_path)
        weights_to_ignore=[]
        if ignore_classifier_weights:
            weights_to_ignore += model.mlp_softmax_classifier.weights
        model.load_weights(initial_model_weights_file_path,
                           weights_to_ignore=weights_to_ignore)

    # train
    nb_epoch = 100

    weights_file_path = weights_file_path_prefix + "cacdi_exp"
    yaml_file = weights_file_path_prefix + "cacdi_exp"
    if use_cnn_as_sequence_to_sequence_encoder:
        weights_file_path += "_cnn"
        yaml_file += "_cnn"
    if pooling_mode:
        weights_file_path += "_" + pooling_mode
        yaml_file += "_" + pooling_mode
    weights_file_path += "_" + optimizer.__class__.__name__

    # consider learning rate,decay and momentum as part of the weight file
    if hasattr(optimizer, 'lr'):
        weights_file_path += "_lr_" + str(round(K.get_value(getattr(optimizer, 'lr')), 6))
    if hasattr(optimizer, 'decay'):
        weights_file_path += "_decay_" + str(round(K.get_value(getattr(optimizer, 'decay')), 6))
    if hasattr(optimizer, 'momentum'):
        weights_file_path += "_momentum_" + str(round(K.get_value(getattr(optimizer, 'momentum')), 6))
    if use_f1_to_early_stop:
        weights_file_path += ".weights.epoch{epoch:03d}.batch{batch:04d}.h5"
    else:
        weights_file_path += ".weights.epoch{epoch:03d}.h5"

    weights_file_path_best = weights_file_path + ".best"

    yaml_file += "_model.yaml"

    if working_directory:
        weights_file_path = os.path.join(working_directory, weights_file_path)
        weights_file_path_best = os.path.join(working_directory,weights_file_path_best)
        yaml_file = os.path.join(working_directory, yaml_file)

    with open(yaml_file, 'w') as file_:
        try:
            yaml = model.to_yaml()
            file_.write(yaml)
        except:
            logger.warning("to_yaml failed")

    callbacks = []
    validation_data = None
    nb_val_samples = None

    if use_f1_to_early_stop:
        callback = EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript(
            val_generator = datasets["cacdi"]["dev"][1],
            val_samples = datasets["cacdi"]["dev"][0],
            filepath = weights_file_path_best,
            save_best_only=True,
            patience=training_patience,
            verbose=1,
            labels_vocabulary = {"UNK":"<UNK>", "label":labels_vocabulary},
            use_per_class_threshold_tuning = use_per_class_threshold_tuning,
            re_try_times=evaluation_re_try_times,
            re_try_waiting_time=evaluation_re_try_waiting_time,
            fail_on_evlauation_failure=fail_on_evlauation_failure,
            working_directory = working_directory,
        )
        callbacks.append(callback)
    else:
        from keras.callbacks import EarlyStopping,ModelCheckpoint
        callbacks.append( EarlyStopping(patience=training_patience))
        callbacks.append(ModelCheckpoint(filepath=weights_file_path_best, save_best_only=True, save_weights_only=True))
        if save_weights_for_each_epoch:
            callbacks.append(ModelCheckpoint(filepath=weights_file_path, save_weights_only=True))
        validation_data = datasets["keras"]["dev"][1]
        nb_val_samples = datasets["keras"]["dev"][0]

    logger.info("Begin training")
    try:
        train_history = model.fit_generator(
            datasets["keras"]["train"][1],
            samples_per_epoch=samples_per_epoch if samples_per_epoch else datasets["keras"]["train"][0],
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=callbacks,
            validation_data=validation_data,
            nb_val_samples=nb_val_samples,
            class_weight=None,
            max_q_size=1)

        logger.info("Training done")

        if use_f1_to_early_stop:
            print("best val-acc: %s" % callback.best_acc)
            logger.info("Loading the best weights...")
            best_epoch = callback.best_epoch
            best_batch = callback.best_batch
            model.load_weights(weights_file_path_best.format(epoch=best_epoch, batch=best_batch))
            optimal_threshold = callback.optimal_threshold
        else:
            val_loss = train_history.history['val_loss']
            print("best val-loss: %s" % val_loss)
            logger.info("Loading the best weights...")
            best_epoch = np.argmin(val_loss)
            model.load_weights(weights_file_path_best.format(epoch=best_epoch))
            optimal_threshold = None

    except (MemoryError, RuntimeError) as e:
        total_recursive_length = 0
        for d in current_row[0]:
            total_recursive_length += d.recursive_length()
        logger.warning("Mini batch too large for memory, with recursive length: %s" %
                       str(total_recursive_length))
        raise

    logger.info("Evaluating...")

    if use_f1_to_early_stop:
        weights_file_path = weights_file_path_best.format(epoch=best_epoch, batch=best_batch)
    else:
        weights_file_path = weights_file_path_best.format(epoch=best_epoch)

    t_before_prediction = time.time()
    optimal_threshold, evaluation_results = EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript.evaluate_with_external_scripts(
        model = model,
        data_generator_with_id = datasets["cacdi"]["test"][1],
        total_nb_samples = datasets["cacdi"]["test"][0],
        filepath = weights_file_path + ".eval",
        optimal_threshold=optimal_threshold,
        labels_vocabulary={"UNK":"<UNK>", "label":labels_vocabulary},
        use_per_class_threshold_tuning=use_per_class_threshold_tuning,
        working_directory=working_directory,
        re_try_times=evaluation_re_try_times,
        re_try_waiting_time=evaluation_re_try_waiting_time)

    delta_ts_prediction = time.time() - t_before_prediction
    print 'Evaluation done, taking %f seconds\n' % (delta_ts_prediction)

    print("optimal threshold %s:" % optimal_threshold)
    print("evaluation results with cacdi script:")
    print(evaluation_results)
    logger.info("Evaluation done")


def infinit_generator(build_generator, mask_value):
    # epoch generator returns one mini-batch/one record of each next
    generator = build_generator()
    i = 0
    while True:
        try:
            n = next(generator)
            i += 1
            # n: could be (x,y,id) or (x,y)
            x = get_sequence_inputs(n[0], mask_value=mask_value)
            # hot fix of y, it is of shape nb_snapshots,1, ICD10_codes
            y = np.squeeze(np.array(n[1]), axis=1)
            yield tuple([x,y] + list(n[2:]))

        except StopIteration:
            if i == 0:
                raise StopIteration("build_generator builds an empty generator")

            i = 0
            generator = build_generator()

def limit_to(limit, generator):
    # if generator is a mini-batch generator, returns a mini-batch each time
    for i, e in enumerate(generator):
        if i >= limit:
            break

        yield e

def extract_snapshot_id(sample, source_index, snapshot_id_index):
    snapshot_features = sample[source_index]
    # TODO: Very dangerous, transform original data...
    #       Purposely remove snapshot id from orignal data. 
    #       But still, it's a side effect so not explicit enough.
    snapshot_id = snapshot_features.pop(snapshot_id_index)
    # sample[source_index] = snapshot_id
    # sample.append(snapshot_id)
    return (snapshot_id, )


def replace_labels(sample, sources, labels_vocabulary):
    snapshot_features = sample[sources.index("features of snapshots")]
    labels_features = np.zeros(len(labels_vocabulary))
    labels = snapshot_features[LABELS_INDEX]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    for label in labels:
        if label != "":
            labels_features[labels_vocabulary[label]] = 1

    snapshot_features[LABELS_INDEX] = labels_features
    return sample


def replace_document_type(sample, sources, document_types_vocabulary):
    documents_features = sample[sources.index("features of documents")]

    # Flattens snapshot structure if needed
    while isinstance(documents_features[0][0], HierarchyLevel):
        documents_features = sum((list(e) for e in documents_features), [])

    if "features of sections" in sources:
        sections_features = sample[sources.index("features of sections")]
        # Flattens snapshot/documents structure if needed
        while isinstance(sections_features[0][0], HierarchyLevel):
            sections_features = sum((list(e) for e in sections_features), [])
    else:
        sections_features = []

    for list_of_features, feature_index in (
        (documents_features, DOCUMENT_TYPE_INDEX),
        (sections_features, SECTION_DOCUMENT_TYPE_INDEX)):

        for features in list_of_features:
            document_type_one_hot = np.zeros(len(document_types_vocabulary))
            document_type = features[feature_index]

            document_type_one_hot[document_types_vocabulary[document_type]] = 1

            features[feature_index] = document_type_one_hot

    return sample


def replace_section_name(sample, sources, section_names_vocabulary):
    sections_features = sample[sources.index("features of sections")]

    # Flattens snapshot/documents structure if needed
    while isinstance(sections_features[0][0], HierarchyLevel):
        sections_features = sum((list(e) for e in sections_features), [])

    for section_features in sections_features:
        section_name_one_hot = np.zeros(len(section_names_vocabulary))
        section_name = section_features[SECTION_NAME_INDEX]

        section_name_one_hot[section_names_vocabulary[section_name]] = 1

        section_features[SECTION_NAME_INDEX] = section_name_one_hot
    return sample


def replace_meta_data(sample, sources, labels_vocabulary,
                      document_types_vocabulary, section_names_vocabulary):

    if "features of sections" in sources:
        sample = replace_section_name(sample, sources, section_names_vocabulary)
    if "features of documents" in sources:
        sample = replace_document_type(sample, sources, document_types_vocabulary)
    sample = replace_labels(sample, sources, labels_vocabulary)

    return sample


def filter_indices(sample, sources, source, indices):
    sample = list(sample)
    sample[sources.index(source)] = _filter_indices(
        sample[sources.index(source)], indices)

    return tuple(sample)


def _filter_indices(data, indices):
    if isinstance(data[0], HierarchyLevel):
        new_data = HierarchyLevel()
        for e in data:
            new_data.append(_filter_indices(e, indices))
        return new_data
    else:
        return HierarchyLevel([data[idx] for idx in indices])

def get_epoch_iterator(data_stream, n):
    global current_row
    data_stream.reset()
    for row in limit_to(n, data_stream.get_epoch_iterator()):
        current_row = row
        yield row


def load_datasets(aggregation_files,
                  min_word_count,
                  batch_size,
                  reduce_length_ratio_over_k_batches=12,
                  unpack=False,
                  debug=False ,
                  maximum_recursive_max_size=500000,
                  maximum_recursive_max_size_for_dev = 5000000,
                  mask_value = -1,
                  use_fixed_minibatch_size = False,
                  ):

    with open(os.path.join(train_source, "labels_vocabulary.pkl")) as f:
        labels_vocabulary = cPickle.load(f)

    with open(os.path.join(train_source, "section_name_vocabulary.pkl")) as f:
        section_names_vocabulary = cPickle.load(f)

    with open(os.path.join(train_source, "document_type_vocabulary.pkl")) as f:
        document_types_vocabulary = cPickle.load(f)


    datasets = defaultdict(dict)
    for partition, source in zip(partition_names, partition_sources):

        dataset = CACDIDatasetWrapper(
            source,
            min_word_count=min_word_count,
            encoding="utf-8",
            aggregation_files=aggregation_files,
            unpack=unpack)

        vocabulary = dataset.get_constrained_vocabulary()

        data_stream = dataset.get_data_stream()

        data_stream = Mapping(
            data_stream, 
            functools.partial(
                replace_meta_data,
                sources=data_stream.sources,
                labels_vocabulary=labels_vocabulary,
                document_types_vocabulary=document_types_vocabulary,
                section_names_vocabulary=section_names_vocabulary))

        if any("sections" in a_f for a_f in aggregation_files):
            data_stream = Mapping(
                data_stream,
                functools.partial(
                    filter_indices,
                    sources=data_stream.sources,
                    source="features of sections",
                    indices=[SECTION_DOCUMENT_TYPE_INDEX, SECTION_NAME_INDEX]))

        if any("documents" in a_f for a_f in aggregation_files):
            data_stream = Mapping(
                data_stream,
                functools.partial(
                    filter_indices,
                    sources=data_stream.sources,
                    source="features of documents",
                    indices=[DOCUMENT_TYPE_INDEX]))

        # Remove section and documents features for now
        data_stream = FilterSources(
            data_stream, ['features', 'features of snapshots'])

        data_stream = Mapping(
            data_stream,
            functools.partial(
                filter_indices,
                sources=data_stream.sources,
                source="features of snapshots",
                indices=[SNAPSHOT_ID_INDEX, LABELS_INDEX]))
        FILTERED_SNAPSHOT_ID_INDEX = 0

        data_stream = Mapping(
            data_stream, 
            functools.partial(
                extract_snapshot_id,
                source_index=data_stream.sources.index("features of snapshots"),
                snapshot_id_index=FILTERED_SNAPSHOT_ID_INDEX),
            add_sources=("ids of snapshots", ))

        # Only do filtering for training dataset
        if use_fixed_minibatch_size:
            data_stream = dataset.mini_batch_stream_by_batch_size(
                data_stream,
                batch_size=batch_size,
                maximum_recursive_max_size=maximum_recursive_max_size if partition == "train" else maximum_recursive_max_size_for_dev,
                reduce_length_ratio_over_k_batches=reduce_length_ratio_over_k_batches
            )
        else:
            data_stream = dataset.mini_batch_stream_by_recursive_max_size(
                data_stream,
                batch_size=batch_size,
                maximum_recursive_max_size=  maximum_recursive_max_size  if partition == "train" else maximum_recursive_max_size_for_dev,
                reduce_length_ratio_over_k_batches=reduce_length_ratio_over_k_batches
                )

        n = dataset.n_samples
        # n should be number of mini-batches
        iterator = infinit_generator(functools.partial(
            get_epoch_iterator, data_stream=data_stream, n=n),
            mask_value=mask_value)

        datasets["cacdi"][partition] = (n, iterator)

        data_stream = FilterSources(
            data_stream, ['features', 'features of snapshots'])

        iterator = infinit_generator(functools.partial(
            get_epoch_iterator, data_stream=data_stream, n=n),
            mask_value=mask_value)

        datasets["keras"][partition] = (n, iterator)

    return datasets, vocabulary,labels_vocabulary
