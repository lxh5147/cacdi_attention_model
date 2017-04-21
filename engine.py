
from keras.engine import Model

from keras import callbacks as cbks
import time
import warnings
from keras.engine.training import (
    standardize_class_weights,
    standardize_weights ,
    generator_queue)


class ModelEx (Model):

    def __init__(self, input, output, on_train_on_batch_failed=None, name=None):
        self.on_train_on_batch_failed = on_train_on_batch_failed
        super(ModelEx, self).__init__(input, output, name)

    def _standardize_user_data(self, x, y,
                               sample_weight=None, class_weight=None,
                               check_batch_dim=True, batch_size=None):
        if not hasattr(self, 'optimizer'):
            raise Exception('You must compile a model before training/testing.'
                            ' Use `model.compile(optimizer, loss)`.')

        # x: nb_samples,source_timesteps and nb_samples, target_timesteps
        # y: nb_samples, target_time_steps
        class_weights = standardize_class_weights(class_weight,
                                                   self.output_names)
        sample_weight = standardize_weights(y,
                                             sample_weight,
                                             class_weights,
                                             self.sample_weight_mode)
        return x, [y], [sample_weight]


    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight={}, max_q_size=10, nb_worker=1, pickle_safe=False):
        '''Fits the model on data generated batch-by-batch by
        a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a generator.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
            samples_per_epoch: integer, number of samples to process before
                going to the next epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            nb_val_samples: only relevant if `validation_data` is a generator.
                number of samples to use from validation generator
                at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up when using process based threading
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        wait_time = 0.01  # in seconds
        epoch = 0

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not nb_val_samples:
            raise Exception('When using a generator for validation data, '
                            'you must specify a value for "nb_val_samples".')

        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks._set_model(callback_model)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': samples_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise Exception('validation_data should be a tuple '
                                '(val_x, val_y, val_sample_weight) '
                                'or (val_x, val_y). Found: ' + str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y, val_sample_weight)
            self.validation_data = val_x + [val_y, val_sample_weights]
        else:
            self.validation_data = None

        # start generator thread storing batches into a queue
        data_gen_queue, _stop,_ = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_worker,
                                                pickle_safe=pickle_safe)

        callback_model.stop_training = False
        while epoch < nb_epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            training_time_total = 0.
            while samples_seen < samples_per_epoch:
                generator_output = None
                while not _stop.is_set():
                    if not data_gen_queue.empty():
                        generator_output = data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                if not hasattr(generator_output, '__len__'):
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                # build batch logs
                batch_logs = {}
                if type(x) is list:
                    batch_size = len(x[0])
                elif type(x) is dict:
                    batch_size = len(list(x.values())[0])
                else:
                    batch_size = len(x)
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)
                # outs = [self.total_loss] + self.metrics_tensors
                try:
                    t_before_prediction = time.time()
                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    delta_ts_prediction = time.time() - t_before_prediction
                    training_time_total += delta_ts_prediction
                    print '\n'
                    print 'Epoch %d updates %d training takes %f seconds\n' % (epoch, batch_index, delta_ts_prediction)

                except Exception as e:
                    if self.on_train_on_batch_failed:
                        try:
                            outs, processed = self.on_train_on_batch_failed(self, x, y,
                                                   sample_weight=sample_weight,
                                                   class_weight=class_weight,
                                                   epoch=epoch,
                                                   batch_index = batch_index,
                                                   exception =e )
                            # update the batch size
                            batch_size = processed
                            batch_logs['size'] = batch_size
                        except:
                            _stop.set()
                            raise
                    else:
                        _stop.set()
                        raise

                if outs is None:
                    batch_logs = {}
                else:
                    if type(outs) != list:
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                # increase batch index only if this batch has been processed, i.e., outputs not empty
                if batch_size > 0:
                    batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen > samples_per_epoch:
                    warnings.warn('Epoch comprised more than '
                                  '`samples_per_epoch` samples, '
                                  'which might affect learning results. '
                                  'Set `samples_per_epoch` correctly '
                                  'to avoid this warning.\n')
                if samples_seen >= samples_per_epoch and do_validation:
                    if val_gen:
                        val_outs = self.evaluate_generator(validation_data,
                                                           nb_val_samples,
                                                           max_q_size=max_q_size)
                    else:
                        # no need for try/except because
                        # data has already been validated
                        val_outs = self.evaluate(val_x, val_y,
                                                 batch_size=batch_size,
                                                 sample_weight=val_sample_weights,
                                                 verbose=0)
                    if type(val_outs) is not list:
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            training_time_total += delta_ts_prediction
            print '\n'
            print 'Epoch %d total %d updates takes %f seconds \n' % (epoch,  batch_index, training_time_total)

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1

            if callback_model.stop_training:
                break

        _stop.set()
        if pickle_safe:
            data_gen_queue.close()
        callbacks.on_train_end()
        return self.history

    # enhanced to filter some weights
    def load_weights(self, filepath, by_name=False, weights_to_ignore=None):
        '''Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        if by_name:
            self.load_weights_from_hdf5_group_by_name(f,
                                                      weights_to_ignore=weights_to_ignore)
        else:
            self.load_weights_from_hdf5_group(f,
                                              weights_to_ignore=weights_to_ignore)

        if hasattr(f, 'close'):
            f.close()


    def load_weights_from_hdf5_group(self, f,
                                     weights_to_ignore=None):
        '''Weight loading is based on layer order in a list
        (matching model.flattened_layers for Sequential models,
        and model.layers for Model class instances), not
        on layer names.
        Layers that have no weights are skipped.
        '''
        import keras.backend as K
        import numpy as np

        if hasattr(self, 'flattened_layers'):
            # Support for legacy Sequential/Merge behavior.
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        if 'nb_layers' in f.attrs:
            # Legacy format.
            nb_layers = f.attrs['nb_layers']
            if nb_layers != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(nb_layers) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + ' layers.')

            for k in range(nb_layers):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                flattened_layers[k].set_weights(weights)
        else:
            # New file format.
            filtered_layers = []
            for layer in flattened_layers:
                weights = layer.weights
                if weights:
                    filtered_layers.append(layer)
            flattened_layers = filtered_layers

            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
            filtered_layer_names = []
            for name in layer_names:
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
                    filtered_layer_names.append(name)
            layer_names = filtered_layer_names
            if len(layer_names) != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(len(layer_names)) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + ' layers.')

            # We batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weight_values = [g[weight_name] for weight_name in weight_names]
                layer = flattened_layers[k]
                symbolic_weights = layer.weights
                if len(weight_values) != len(symbolic_weights):
                    raise Exception('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')
                if layer.__class__.__name__ == 'Convolution1D':
                    # This is for backwards compatibility with
                    # the old Conv1D weights format.
                    w = weight_values[0]
                    shape = w.shape
                    if shape[:2] != (layer.filter_length, 1) or shape[3] != layer.nb_filter:
                        # Legacy shape: (self.nb_filter, input_dim, self.filter_length, 1)
                        assert shape[0] == layer.nb_filter and shape[2:] == (layer.filter_length, 1)
                        w = np.transpose(w, (2, 3, 1, 0))
                        weight_values[0] = w
                for weight, value in zip(symbolic_weights, weight_values):
                # ignore to update this weight if applicable
                    if weights_to_ignore:
                        if weight in weights_to_ignore:
                            continue
                    weight_value_tuples.append((weight, value))

            K.batch_set_value(weight_value_tuples)


    def load_weights_from_hdf5_group_by_name(self, f,
                                             weights_to_ignore=None):
        ''' Name-based weight loading
        (instead of topological weight loading).
        Layers that have no matching name are skipped.
        '''
        import keras.backend as K

        if hasattr(self, 'flattened_layers'):
            # Support for legacy Sequential/Merge behavior.
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        if 'nb_layers' in f.attrs:
            raise Exception('The weight file you are trying to load is' +
                            ' in a legacy format that does not support' +
                            ' name-based weight loading.')
        else:
            # New file format.
            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

            # Reverse index of layer name to list of layers with name.
            index = {}
            for layer in flattened_layers:
                if layer.name:
                    index.setdefault(layer.name, []).append(layer)

            # We batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weight_values = [g[weight_name] for weight_name in weight_names]

                for layer in index.get(name, []):
                    symbolic_weights = layer.weights
                    if len(weight_values) != len(symbolic_weights):
                        raise Exception('Layer #' + str(k) +
                                        ' (named "' + layer.name +
                                        '") expects ' +
                                        str(len(symbolic_weights)) +
                                        ' weight(s), but the saved weights' +
                                        ' have ' + str(len(weight_values)) +
                                        ' element(s).')
                    # Set values.
                    for i in range(len(weight_values)):
                        # ignore to update this weight if applicable
                        if weights_to_ignore:
                            if symbolic_weights[i] in weights_to_ignore:
                                continue
                        weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
            K.batch_set_value(weight_value_tuples)
